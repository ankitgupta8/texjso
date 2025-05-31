import json
import requests
import csv
from io import StringIO
import google.generativeai as genai
import re

def preprocess_latex(text):
    """
    Preprocess text containing LaTeX equations to make it JSON-safe while preserving the equations
    """
    def escape_latex(match):
        latex = match.group(0)
        # Preserve the dollar signs and escape necessary characters
        return latex.replace('\\', '\\\\').replace('"', '\\"')
    
    # Process both inline and display math mode LaTeX
    text = re.sub(r'\$\$.*?\$\$|\$.*?\$', escape_latex, text, flags=re.DOTALL)
    return text

def call_gemini_api(text_input, api_key):
    """
    Call Gemini API to convert text to quiz format
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

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

    Important instructions:
    - generate as many questions as possible
    -correct answer should be verified and most of the time answer will be in text itself 
    1. Create as many questions as possible from the content (minimum 10 questions)
    2. Group all questions into a single chapter instead of creating multiple chapters
    3. dont create a new chapter
    4. Make sure each question has detailed explanations
    5. Ensure all options are relevant to the question
    6. Verify that all options are unique and not duplicate/similar
    7. Each option must be factually different from others
    8. The correct answer must be exactly matching one of the given options
    9. Options should be clear and unambiguous
    10. No two options should be the same in text. if same then change options.
    11. Review each option pair and rephrase if they:
        - Have similar wording or meaning
        - Could be interpreted as synonyms
        - Present the same concept differently
    12. Ensure each option represents a distinct and unique choice
    13. For mathematical content:
        - Use LaTeX equations enclosed in $ for inline math or $$ for display math
        - Make sure to properly format all mathematical expressions in LaTeX
        - Keep LaTeX equations intact and properly escaped in the response"""

    try:
        full_prompt = f"{system_prompt}\n\nText to convert:\n{text_input}"
        response = model.generate_content(full_prompt)

        if not response.text:
            return None, "Empty response from API"

        try:
            # Preprocess the response text to handle LaTeX equations
            text = response.text.strip()
            
            # Remove any markdown formatting that might be present
            text = re.sub(r'```json|```', '', text)
            
            # Find the first { and last } to extract JSON
            start = text.find('{')
            end = text.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = text[start:end]
                # Clean up whitespace while preserving LaTeX
                json_str = re.sub(r'\s+', ' ', json_str)
                # Preprocess LaTeX equations
                json_str = preprocess_latex(json_str)
                
                # Parse JSON
                quiz_json = json.loads(json_str)
                
                # Verify that LaTeX equations are preserved
                def verify_latex(obj):
                    if isinstance(obj, dict):
                        return {k: verify_latex(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [verify_latex(item) for item in obj]
                    elif isinstance(obj, str):
                        # Ensure LaTeX delimiters are preserved
                        if '$' in obj:
                            return obj
                        return obj
                    return obj
                
                quiz_json = verify_latex(quiz_json)
                return quiz_json, None
            else:
                return None, "No valid JSON found in response"
                
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try additional cleanup
            try:
                # Additional cleanup attempt
                text = re.sub(r'[^\x20-\x7E\n]', '', text)  # Remove non-printable characters
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = text[start:end]
                    json_str = preprocess_latex(json_str)
                    quiz_json = json.loads(json_str)
                    quiz_json = verify_latex(quiz_json)
                    return quiz_json, None
            except:
                return None, f"Failed to parse JSON response: {str(e)}"
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"

    except Exception as e:
        return None, f"API Error: {str(e)}"

def format_json(json_data):
    """
    Format JSON data with proper indentation
    """
    return json.dumps(json_data, indent=2)

def convert_to_csv(json_data):
    """
    Convert quiz JSON to CSV format
    """
    output = StringIO()
    csv_writer = csv.writer(output)

    # Write header
    csv_writer.writerow(['Question', 'Option A', 'Option B', 'Option C', 'Option D', 'Correct Answer', 'Explanation'])

    # Extract questions from all chapters
    for chapter in json_data['chapters']:
        for question in chapter['quizQuestions']:
            row = [
                question['question'],
                question['options'][0] if len(question['options']) > 0 else '',
                question['options'][1] if len(question['options']) > 1 else '',
                question['options'][2] if len(question['options']) > 2 else '',
                question['options'][3] if len(question['options']) > 3 else '',
                question['correctAnswer'],
                question['explanation']
            ]
            csv_writer.writerow(row)

    return output.getvalue()
