import json
import requests
import csv
from io import StringIO
import google.generativeai as genai
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuizGeneratorError(Exception):
    """Custom exception for quiz generator errors"""
    pass

def preprocess_latex(text: str) -> str:
    """
    Preprocess text containing LaTeX equations to make it JSON-safe while preserving the equations
    """
    if not text or not isinstance(text, str):
        return text
    
    def escape_latex(match):
        latex = match.group(0)
        # More comprehensive escaping for JSON safety
        escaped = latex.replace('\\', '\\\\')
        escaped = escaped.replace('"', '\\"')
        escaped = escaped.replace('\n', '\\n')
        escaped = escaped.replace('\r', '\\r')
        escaped = escaped.replace('\t', '\\t')
        return escaped
    
    try:
        # Process both inline and display math mode LaTeX with more robust patterns
        # Handle nested brackets and escaped delimiters
        patterns = [
            r'\$\$(?:[^$]|\$(?!\$))*\$\$',  # Display math
            r'\$(?:[^$\n]|\\.)*\$',         # Inline math
            r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}',  # LaTeX environments
            r'\\left[(\[{|].*?\\right[)\]}|]',      # Balanced delimiters
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, escape_latex, text, flags=re.DOTALL)
        
        return text
    except Exception as e:
        logger.warning(f"LaTeX preprocessing failed: {e}")
        return text

def validate_json_structure(data: Dict) -> bool:
    """
    Validate that the JSON has the expected quiz structure
    """
    try:
        if not isinstance(data, dict):
            return False
        
        required_keys = ['subject', 'chapters']
        if not all(key in data for key in required_keys):
            return False
        
        if not isinstance(data['chapters'], list) or len(data['chapters']) == 0:
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
                
                required_q_keys = ['question', 'options', 'correctAnswer', 'explanation']
                if not all(key in question for key in required_q_keys):
                    return False
                
                if not isinstance(question['options'], list) or len(question['options']) < 2:
                    return False
                
                # Validate that correct answer exists in options
                if question['correctAnswer'] not in question['options']:
                    logger.warning(f"Correct answer not found in options: {question['correctAnswer']}")
        
        return True
    except Exception as e:
        logger.error(f"JSON validation failed: {e}")
        return False

def clean_json_response(text: str) -> str:
    """
    Clean and extract JSON from API response with multiple fallback strategies
    """
    if not text:
        return ""
    
    # Remove common markdown formatting
    text = re.sub(r'```(?:json)?\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Try to find JSON boundaries
    strategies = [
        # Strategy 1: Find complete JSON object
        lambda t: t[t.find('{'):t.rfind('}') + 1] if '{' in t and '}' in t else None,
        
        # Strategy 2: Find JSON array if object fails
        lambda t: t[t.find('['):t.rfind(']') + 1] if '[' in t and ']' in t else None,
        
        # Strategy 3: Clean up whitespace and try again
        lambda t: re.sub(r'\s+', ' ', t),
        
        # Strategy 4: Remove control characters
        lambda t: ''.join(char for char in t if ord(char) >= 32 or char in '\n\r\t'),
    ]
    
    for strategy in strategies:
        try:
            result = strategy(text)
            if result and result.strip():
                return result.strip()
        except Exception as e:
            logger.debug(f"Cleaning strategy failed: {e}")
            continue
    
    return text

def call_gemini_api(text_input: str, api_key: str, max_retries: int = 3) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Call Gemini API to convert text to quiz format with robust error handling
    """
    if not text_input or not text_input.strip():
        return None, "Empty input text provided"
    
    if not api_key:
        return None, "API key not provided"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        return None, f"Failed to configure Gemini API: {str(e)}"

    system_prompt = """Convert the following text into a quiz format JSON with the following structure:
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
                        "explanation": "Detailed explanation of the answer"
                    }
                ]
            }
        ]
    }

    CRITICAL REQUIREMENTS:
    1. Generate as many questions as possible from the content (minimum 10 questions)
    2. Group all questions into a single chapter instead of creating multiple chapters
    3. Make sure each question has detailed explanations
    4. Ensure all options are relevant and unique
    5. The correct answer must EXACTLY match one of the options
    6. For mathematical content, use proper LaTeX formatting:
       - Inline equations: $...$
       - Display equations: $$...$$
       - Proper escaping for JSON compatibility
    7. Validate that no two options are identical or too similar
    8. Ensure proper JSON formatting with no syntax errors
    9. Return ONLY valid JSON, no additional text or explanations
    """

    for attempt in range(max_retries):
        try:
            full_prompt = f"{system_prompt}\n\nText to convert:\n{text_input}"
            
            # Add retry delay for rate limiting
            if attempt > 0:
                time.sleep(2 ** attempt)  # Exponential backoff
            
            response = model.generate_content(
                full_prompt,
                generation_config={
                    'temperature': 0.1,  # Lower temperature for more consistent JSON
                    'max_output_tokens': 8192,
                }
            )

            if not response or not response.text:
                logger.warning(f"Empty response from API (attempt {attempt + 1})")
                continue

            # Clean and preprocess the response
            raw_text = response.text.strip()
            cleaned_text = clean_json_response(raw_text)
            processed_text = preprocess_latex(cleaned_text)
            
            # Multiple JSON parsing attempts
            json_candidates = [
                processed_text,
                cleaned_text,
                raw_text,
            ]
            
            for candidate in json_candidates:
                if not candidate:
                    continue
                    
                try:
                    quiz_json = json.loads(candidate)
                    
                    # Validate structure
                    if not validate_json_structure(quiz_json):
                        logger.warning("Invalid JSON structure")
                        continue
                    
                    # Post-process to ensure data integrity
                    quiz_json = post_process_quiz_data(quiz_json)
                    
                    logger.info(f"Successfully parsed JSON on attempt {attempt + 1}")
                    return quiz_json, None
                    
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parse failed for candidate: {str(e)}")
                    continue
                except Exception as e:
                    logger.debug(f"Unexpected error parsing candidate: {str(e)}")
                    continue
            
            logger.warning(f"All JSON parsing attempts failed on attempt {attempt + 1}")
            
        except Exception as e:
            logger.error(f"API call failed on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return None, f"API Error after {max_retries} attempts: {str(e)}"
    
    return None, f"Failed to generate valid quiz after {max_retries} attempts"

def post_process_quiz_data(quiz_json: Dict) -> Dict:
    """
    Post-process quiz data to fix common issues
    """
    try:
        # Ensure subject exists
        if 'subject' not in quiz_json or not quiz_json['subject']:
            quiz_json['subject'] = "Generated Quiz"
        
        # Process each chapter
        for chapter in quiz_json.get('chapters', []):
            if 'chapterName' not in chapter or not chapter['chapterName']:
                chapter['chapterName'] = "Chapter 1"
            
            # Process each question
            for i, question in enumerate(chapter.get('quizQuestions', [])):
                # Ensure question has required fields
                if 'question' not in question or not question['question']:
                    question['question'] = f"Question {i + 1}"
                
                if 'options' not in question or not isinstance(question['options'], list):
                    question['options'] = ["Option A", "Option B", "Option C", "Option D"]
                
                # Ensure we have at least 2 options
                while len(question['options']) < 2:
                    question['options'].append(f"Option {len(question['options']) + 1}")
                
                # Fix correct answer if it doesn't match any option
                if 'correctAnswer' not in question or question['correctAnswer'] not in question['options']:
                    question['correctAnswer'] = question['options'][0]
                
                if 'explanation' not in question or not question['explanation']:
                    question['explanation'] = "No explanation provided."
                
                # Remove duplicate options
                unique_options = []
                seen = set()
                for option in question['options']:
                    if option.strip().lower() not in seen:
                        unique_options.append(option)
                        seen.add(option.strip().lower())
                
                # Ensure we still have the correct answer after deduplication
                if question['correctAnswer'] not in unique_options:
                    unique_options[0] = question['correctAnswer']
                
                question['options'] = unique_options
        
        return quiz_json
    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        return quiz_json

def format_json(json_data: Dict, indent: int = 2) -> str:
    """
    Format JSON data with proper indentation and error handling
    """
    try:
        return json.dumps(json_data, indent=indent, ensure_ascii=False)
    except Exception as e:
        logger.error(f"JSON formatting failed: {e}")
        return str(json_data)

def convert_to_csv(json_data: Dict) -> str:
    """
    Convert quiz JSON to CSV format with robust error handling
    """
    if not json_data or not isinstance(json_data, dict):
        raise QuizGeneratorError("Invalid JSON data for CSV conversion")
    
    try:
        output = StringIO()
        csv_writer = csv.writer(output, quoting=csv.QUOTE_ALL)

        # Write header
        headers = ['Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Answer', 'Explanation']
        csv_writer.writerow(headers)

        # Extract questions from all chapters
        chapters = json_data.get('chapters', [])
        if not chapters:
            raise QuizGeneratorError("No chapters found in quiz data")
        
        question_count = 0
        for chapter in chapters:
            questions = chapter.get('quizQuestions', [])
            for question in questions:
                try:
                    # Safely extract data with defaults
                    q_text = question.get('question', 'No question')
                    options = question.get('options', [])
                    correct = question.get('correctAnswer', 'No answer')
                    explanation = question.get('explanation', 'No explanation')
                    
                    # Ensure we have at least 4 options (pad with empty strings)
                    while len(options) < 4:
                        options.append('')
                    
                    row = [
                        q_text,
                        options[0] if len(options) > 0 else '',
                        options[1] if len(options) > 1 else '',
                        options[2] if len(options) > 2 else '',
                        options[3] if len(options) > 3 else '',
                        correct,
                        explanation
                    ]
                    csv_writer.writerow(row)
                    question_count += 1
                except Exception as e:
                    logger.warning(f"Skipping malformed question: {e}")
                    continue
        
        if question_count == 0:
            raise QuizGeneratorError("No valid questions found for CSV conversion")
        
        logger.info(f"Successfully converted {question_count} questions to CSV")
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"CSV conversion failed: {e}")
        raise QuizGeneratorError(f"CSV conversion failed: {str(e)}")

def validate_latex_in_quiz(quiz_data: Dict) -> List[str]:
    """
    Validate LaTeX formatting in quiz data and return warnings
    """
    warnings = []
    
    def check_latex_balance(text: str) -> List[str]:
        issues = []
        if not text:
            return issues
        
        # Check for unbalanced dollar signs
        single_dollars = text.count('$') - 2 * text.count('$$')
        if single_dollars % 2 != 0:
            issues.append("Unbalanced single dollar signs ($)")
        
        # Check for unbalanced braces
        open_braces = text.count('{')
        close_braces = text.count('}')
        if open_braces != close_braces:
            issues.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")
        
        return issues
    
    try:
        for chapter in quiz_data.get('chapters', []):
            for i, question in enumerate(chapter.get('quizQuestions', [])):
                q_num = i + 1
                
                # Check question text
                issues = check_latex_balance(question.get('question', ''))
                for issue in issues:
                    warnings.append(f"Question {q_num}: {issue}")
                
                # Check options
                for j, option in enumerate(question.get('options', [])):
                    issues = check_latex_balance(option)
                    for issue in issues:
                        warnings.append(f"Question {q_num}, Option {j+1}: {issue}")
                
                # Check explanation
                issues = check_latex_balance(question.get('explanation', ''))
                for issue in issues:
                    warnings.append(f"Question {q_num} explanation: {issue}")
    
    except Exception as e:
        warnings.append(f"LaTeX validation failed: {e}")
    
    return warnings

# Example usage and testing function
def test_quiz_generator():
    """
    Test function to validate the quiz generator
    """
    sample_text = """
    The Pythagorean theorem states that in a right triangle, 
    the square of the hypotenuse equals the sum of squares of the other two sides.
    Mathematically, this can be expressed as $a^2 + b^2 = c^2$.
    
    Another important formula is the quadratic formula: 
    $$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$
    """
    
    # This would require a real API key
    api_key = "your-api-key-here"
    
    try:
        quiz_data, error = call_gemini_api(sample_text, api_key)
        
        if error:
            print(f"Error: {error}")
            return False
        
        if quiz_data:
            print("Quiz generated successfully!")
            
            # Validate LaTeX
            latex_warnings = validate_latex_in_quiz(quiz_data)
            if latex_warnings:
                print("LaTeX warnings:")
                for warning in latex_warnings:
                    print(f"  - {warning}")
            
            # Convert to formats
            json_output = format_json(quiz_data)
            csv_output = convert_to_csv(quiz_data)
            
            print(f"JSON length: {len(json_output)}")
            print(f"CSV length: {len(csv_output)}")
            
            return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_quiz_generator()
