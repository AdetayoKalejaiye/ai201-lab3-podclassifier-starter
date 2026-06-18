import json
import os
import re
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.
    
    Instructs the LLM to categorize the podcast format and emit a structured JSON response.
    """
    # Define baseline task instructions and label schema
    prompt_lines = [
        "You are classifying podcast episodes by their format. Classify the target episode ",
        "into exactly one of these four valid labels:\n",
        "- interview: a conversation between a host and one or more guests",
        "- solo: a single host speaking from memory, experience, or opinion — no guests, no assembled external sources",
        "- panel: multiple guests with roughly equal speaking time, often debating or discussing a topic together",
        "- narrative: a story assembled from external sources — interviews, archival audio, reporting — with a clear narrative arc\n",
        "Your response must be a raw, valid JSON object containing exactly two keys:",
        '  "label": (string) Must be one of the four labels exactly as specified above, or "unknown" if completely unclassifiable.',
        '  "reasoning": (string) A concise, one-sentence explanation of why this label applies.\n',
        "Do not include any greeting, preamble, markdown code blocks, or postscript wrapper outside of the JSON block.",
        "\n--- START OF EXAMPLES ---"
    ]

    # Append labeled training examples dynamically if present
    if labeled_examples:
        for idx, example in enumerate(labeled_examples, 1):
            title = example.get("title", "No Title")
            desc = example.get("description", "No description available.")
            label = example.get("label", "unknown")
            
            prompt_lines.append(f"\nExample #{idx}:")
            prompt_lines.append(f"Title: {title}")
            prompt_lines.append(f"Description: {desc}")
            
            example_json = {
                "label": label,
                "reasoning": f"Matches standard features expected for an {label} format episode."
            }
            prompt_lines.append(f"Output:\n{json.dumps(example_json, indent=2)}\n")
            prompt_lines.append("---")
    else:
        prompt_lines.append("\n(No training examples provided. Classify using the rules above.)")

    prompt_lines.append("\n--- END OF EXAMPLES ---")
    
    # Append the target episode payload to evaluate
    prompt_lines.append("\n### Target Episode to Classify ###")
    prompt_lines.append(f"Description: {description if description else '[Empty Description]'}\n")
    prompt_lines.append("Output your completed JSON object below:")

    return "\n".join(prompt_lines)


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    Steps:
      1. Call build_few_shot_prompt() to construct the prompt
      2. Send it to the LLM via _client.chat.completions.create()
      3. Parse the response to extract a label and reasoning
      4. Validate the label — if it's not in VALID_LABELS, set it to "unknown"
      5. Return a dict with "label" and "reasoning" keys
    """
    try:
        # Step 1 — Build the prompt
        prompt = build_few_shot_prompt(labeled_examples, description)
        
        # Step 2 — Send request to LLM
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a precise classification utility that communicates strictly in structured JSON."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=300,
            temperature=0.0  # Kept low for deterministic and compliant output structural formatting
        )
        
        raw_content = response.choices[0].message.content.strip()
        
        # Step 3 — Parse the response safely
        # Strip potential markdown fences if present (e.g. ```json ... ```)
        cleaned_content = raw_content
        if cleaned_content.startswith("```"):
            cleaned_content = re.sub(r"```$", "", cleaned_content).strip()

        parsed_data = json.loads(cleaned_content)
        
        extracted_label = str(parsed_data.get("label", "unknown")).strip().lower()
        reasoning = parsed_data.get("reasoning", "No explanation provided by classifier.")
        
        # Step 4 — Validate the extracted label
        final_label = "unknown"
        for valid in VALID_LABELS:
            if valid.lower() == extracted_label:
                final_label = valid
                break

        return {
            "label": final_label,
            "reasoning": reasoning
        }

    except Exception as e:
        # Step 5 — Handle errors gracefully to prevent pipeline crashes
        return {
            "label": "unknown",
            "reasoning": f"Graceful processing fallback activated. Internal error: {str(e)}"
        }