import json
import requests
import csv
from io import StringIO
import google.generativeai as genai
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
import time

# ─────────────────────────────── logging ──────────────────────────────── #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────── errors ──────────────────────────────── #
class QuizGeneratorError(Exception):
    """Custom exception for quiz-generator errors."""
    pass

# ────────────────────────── helper: LaTeX validator ───────────────────── #
def validate_and_warn_latex(text: str) -> List[str]:
    """
    *Passive* check — never mutates the string.
    Returns a list of warnings about obviously malformed LaTeX delimiters.
    """
    issues = []
    if not text:
        return issues

    # unmatched single '$' (ignore '$$ … $$')
    singles = text.count('$') - 2 * text.count('$$')
    if singles % 2:
        issues.append("Unbalanced single dollar signs ($)")

    # unmatched braces
    opens, closes = text.count('{'), text.count('}')
    if opens != closes:
        issues.append(f"Unbalanced braces: {{={opens}}, }}={closes}")

    return issues

# ─────────────────────────── helper: JSON shape ───────────────────────── #
def validate_json_structure(data: Dict) -> bool:
    """Validate that the JSON has the expected quiz structure."""
    try:
        if not isinstance(data, dict):
            return False

        required_keys = ['subject', 'chapters']
        if not all(key in data for key in required_keys):
            return False

        if not isinstance(data['chapters'], list) or not data['chapters']:
            return False

        for chapter in data['chapters']:
            if not isinstance(chapter, dict):
                return False
            if 'chapterName' not in chapter or 'quizQuestions' not in chapter:
                return False
            if not isinstance(chapter['quizQuestions'], list):
                return False

            for question in chapter['quizQuestions']:
                if not isinstance(question, dict):
                    return False
                required_q_keys = ['question', 'options',
                                   'correctAnswer', 'explanation']
                if not all(key in question for key in required_q_keys):
                    return False
                if not isinstance(question['options'], list) or len(question['options']) < 2:
                    return False
                if question['correctAnswer'] not in question['options']:
                    logger.warning(
                        "Correct answer not found in options: %s",
                        question['correctAnswer']
                    )
        return True
    except Exception as e:
        logger.error("JSON validation failed: %s", e)
        return False

# ─────────────────────── helper: clean model output ───────────────────── #
def clean_json_response(text: str) -> str:
    """
    Strip markdown fences & obvious noise; return best-guess JSON string.
    """
    if not text:
        return ""

    # Remove common markdown code fences
    text = re.sub(r'```(?:json)?\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text)

    text = text.strip()

    # Heuristics
    strategies = [
        # full object
        lambda t: t[t.find('{'):t.rfind('}') + 1] if '{' in t and '}' in t else "",
        # full array
        lambda t: t[t.find('['):t.rfind(']') + 1] if '[' in t and ']' in t else "",
        # collapse whitespace
        lambda t: re.sub(r'\s+', ' ', t),
        # drop control chars
        lambda t: ''.join(ch for ch in t if ord(ch) >= 32 or ch in '\n\r\t'),
    ]

    for strat in strategies:
        try:
            candidate = strat(text)
            if candidate.strip():
                return candidate.strip()
        except Exception:
            continue
    return text

# ────────────────────────────── core: Gemini ──────────────────────────── #
def call_gemini_api(text_input: str, api_key: str,
                    max_retries: int = 3) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Call Gemini, clean/validate/parse JSON, return Python dict on success.
    """
    if not text_input or not text_input.strip():
        return None, "Empty input text provided"
    if not api_key:
        return None, "API key not provided"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        return None, f"Failed to configure Gemini API: {e}"

    system_prompt = """
Convert the following text into a quiz-format JSON:

{
  "subject": "Subject name based on content",
  "chapters": [
    {
      "chapterName": "Chapter name based on content",
      "quizQuestions": [
        {
          "question": "Question text",
          "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
          "correctAnswer": "Correct option",
          "explanation": "Detailed explanation"
        }
      ]
    }
  ]
}

CRITICAL REQUIREMENTS
1. ≥10 questions, all in ONE chapter
2. Detailed explanations
3. Unique, relevant options
4. correctAnswer must EXACTLY match one option
5. Math content in proper LaTeX: $…$ for inline, $$…$$ for display
6. Return ONLY valid JSON — no prose
"""

    for attempt in range(max_retries):
        try:
            if attempt:
                time.sleep(2 ** attempt)         # exponential back-off

            response = model.generate_content(
                f"{system_prompt}\n\nText to convert:\n{text_input}",
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 8192
                }
            )

            if not response or not response.text:
                logger.warning("Empty API response (attempt %d)", attempt + 1)
                continue

            raw_text = response.text.strip()
            cleaned_text = clean_json_response(raw_text)

            # passive LaTeX sanity check
            latex_issues = validate_and_warn_latex(cleaned_text)
            for issue in latex_issues:
                logger.warning("LaTeX issue: %s", issue)

            # Try to parse (no extra escaping!)
            for candidate in (cleaned_text, raw_text):
                if not candidate:
                    continue
                try:
                    quiz_json = json.loads(candidate)

                    if not validate_json_structure(quiz_json):
                        logger.warning("Invalid JSON structure")
                        continue

                    quiz_json = post_process_quiz_data(quiz_json)
                    logger.info("Parsed JSON on attempt %d", attempt + 1)
                    return quiz_json, None

                except json.JSONDecodeError as e:
                    logger.debug("JSON decode error: %s", e)
                    continue

            logger.warning("Parsing failed on attempt %d", attempt + 1)

        except Exception as e:
            logger.error("Attempt %d failed: %s", attempt + 1, e)
            if attempt == max_retries - 1:
                return None, f"API Error after {max_retries} attempts: {e}"

    return None, f"Failed to generate valid quiz after {max_retries} attempts"

# ──────────────────── post-processing & other utilities ───────────────── #
def post_process_quiz_data(quiz_json: Dict) -> Dict:
    """Fix common omissions & ensure data integrity."""
    try:
        quiz_json.setdefault('subject', "Generated Quiz")

        for chap_idx, chapter in enumerate(quiz_json.get('chapters', []), 1):
            chapter.setdefault('chapterName', f"Chapter {chap_idx}")

            for q_idx, q in enumerate(chapter.get('quizQuestions', []), 1):
                q.setdefault('question', f"Question {q_idx}")
                q.setdefault('options', ["Option A", "Option B"])
                while len(q['options']) < 2:
                    q['options'].append(f"Option {len(q['options']) + 1}")
                if q.get('correctAnswer') not in q['options']:
                    q['correctAnswer'] = q['options'][0]
                q.setdefault('explanation', "No explanation provided.")

                # dedupe options (case-insensitive)
                unique, seen = [], set()
                for opt in q['options']:
                    key = opt.strip().lower()
                    if key not in seen:
                        seen.add(key)
                        unique.append(opt)
                if q['correctAnswer'] not in unique:
                    unique.insert(0, q['correctAnswer'])
                q['options'] = unique

        return quiz_json
    except Exception as e:
        logger.error("Post-processing failed: %s", e)
        return quiz_json

def format_json(json_data: Dict, indent: int = 2) -> str:
    try:
        return json.dumps(json_data, indent=indent, ensure_ascii=False)
    except Exception as e:
        logger.error("JSON formatting failed: %s", e)
        return str(json_data)

def convert_to_csv(json_data: Dict) -> str:
    """Convert validated quiz JSON → CSV (one row per question)."""
    if not isinstance(json_data, dict):
        raise QuizGeneratorError("Invalid JSON data for CSV conversion")

    output = StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    headers = ['Question', 'Option A', 'Option B',
               'Option C', 'Option D', 'Correct Answer', 'Explanation']
    writer.writerow(headers)

    question_count = 0
    for chapter in json_data.get('chapters', []):
        for q in chapter.get('quizQuestions', []):
            options = q.get('options', []) + [''] * 4
            row = [
                q.get('question', 'No question'),
                options[0], options[1], options[2], options[3],
                q.get('correctAnswer', ''),
                q.get('explanation', '')
            ]
            writer.writerow(row)
            question_count += 1

    if not question_count:
        raise QuizGeneratorError("No questions to write")

    logger.info("Converted %d questions to CSV", question_count)
    return output.getvalue()

# ──────────────────────────── quick tester ────────────────────────────── #
def test_quiz_generator():
    sample_text = """
The Pythagorean theorem states that in a right triangle
the square of the hypotenuse equals the sum of squares of the other two sides.
Mathematically: $a^2 + b^2 = c^2$.

The quadratic formula:
$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$
"""

    api_key = "your-api-key-here"
    quiz, err = call_gemini_api(sample_text, api_key)

    if err:
        print("Error:", err)
        return False

    print("Quiz generated!")
    print("LaTeX warnings:", validate_and_warn_latex(json.dumps(quiz)))

    print("JSON preview:", format_json(quiz)[:200], "…")
    print("CSV preview:", convert_to_csv(quiz)[:200], "…")
    return True

if __name__ == "__main__":
    test_quiz_generator()
