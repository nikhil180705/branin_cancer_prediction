"""
report_generator.py - AI-Powered Medical Report Generation

Generates structured medical-style reports from brain tumor predictions
using an open-source LLM (Qwen2.5-0.5B-Instruct) with a template-based
fallback for reliability.

Pipeline: Prediction → Structured JSON → LLM Prompt → Medical Report
"""

import os
import json
import traceback

# ── Template-Based Report Generator (Always Available) ───────────────────

# Recommendation mappings based on tumor type and risk
_RECOMMENDATIONS = {
    'glioma': {
        'High': [
            'It is strongly advised to consult a neurologist or oncologist immediately.',
            'Follow-up imaging (contrast-enhanced MRI) and diagnostic tests should be scheduled.',
            'A biopsy may be recommended to determine the exact grade of the glioma.',
            'Avoid stress and maintain a healthy lifestyle as advised by a medical professional.',
            'Discuss treatment options including surgery, radiation, and chemotherapy with your care team.'
        ],
        'Medium': [
            'Consultation with a neurologist is recommended in the near term.',
            'Follow-up MRI imaging should be scheduled within the next few weeks.',
            'Monitor for symptoms such as headaches, seizures, or cognitive changes.',
            'Maintain a balanced diet and adequate rest.',
            'Discuss monitoring and treatment planning with a specialist.'
        ],
        'Low': [
            'Regular monitoring with periodic MRI scans is recommended.',
            'Consult a neurologist for a professional evaluation.',
            'Report any new neurological symptoms promptly.',
            'Maintain overall health through balanced nutrition and exercise.'
        ]
    },
    'meningioma': {
        'High': [
            'Consultation with a neurosurgeon is strongly recommended.',
            'Follow-up imaging should be scheduled promptly.',
            'Surgical evaluation may be warranted depending on the tumor location and symptoms.',
            'Monitor for symptoms such as vision changes, headaches, or weakness.',
            'Discuss treatment options with a qualified medical professional.'
        ],
        'Medium': [
            'Schedule an appointment with a neurologist for further evaluation.',
            'Follow-up MRI imaging is recommended within the next 1-2 months.',
            'Meningiomas are commonly slow-growing; monitor for symptom changes.',
            'Maintain regular medical checkups and report any new symptoms.'
        ],
        'Low': [
            'Regular monitoring with periodic imaging is advised.',
            'Most meningiomas are benign and slow-growing.',
            'Consult a neurologist if new symptoms develop.',
            'Maintain a healthy lifestyle and regular checkups.'
        ]
    },
    'pituitary': {
        'High': [
            'Consultation with an endocrinologist and neurosurgeon is recommended.',
            'Hormonal blood tests should be performed to assess pituitary function.',
            'Follow-up imaging is recommended to monitor tumor progression.',
            'Report any vision changes, hormonal symptoms, or headaches.',
            'Discuss treatment options which may include medication or surgery.'
        ],
        'Medium': [
            'Schedule an evaluation with an endocrinologist.',
            'Hormonal assessment through blood work is advised.',
            'Follow-up MRI should be scheduled in the near future.',
            'Monitor for symptoms related to hormonal imbalances.'
        ],
        'Low': [
            'Regular monitoring is recommended with periodic imaging.',
            'Pituitary tumors are often benign and manageable.',
            'Consult an endocrinologist if hormonal symptoms appear.',
            'Maintain regular health checkups.'
        ]
    },
    'notumor': {
        'Low': [
            'No immediate medical intervention appears necessary based on this scan.',
            'Continue routine health checkups as recommended by your physician.',
            'If symptoms persist, consult a neurologist for further evaluation.',
            'Maintain a healthy lifestyle.'
        ]
    }
}

_PRECAUTIONS = {
    'High': 'Based on the current assessment, the patient should take precautions seriously and seek medical consultation without delay. Timely intervention is critical for optimal outcomes.',
    'Medium': 'The patient should schedule a medical consultation in the near term. While the condition warrants attention, careful monitoring and professional guidance can help manage the situation effectively.',
    'Low': 'The current assessment indicates a lower level of concern. However, maintaining regular medical follow-ups and staying attentive to any changes in symptoms remains important.'
}

_URGENCY = {
    'High': 'The urgency of the recommended actions is high, and following these precautions is strongly advised.',
    'Medium': 'The recommended actions should be followed within a reasonable timeframe to ensure proper monitoring.',
    'Low': 'The recommendations can be followed at the next regular medical visit unless new symptoms develop.'
}


def generate_report_template(prediction_data):
    """
    Generate a structured medical report using templates.
    This is the reliable fallback that always works.
    
    Args:
        prediction_data (dict): {
            'tumor_detected': bool,
            'tumor_type': str,
            'confidence': float (0-1),
            'tumor_size': str,
            'risk_level': str,
            'probabilities': dict (optional)
        }
    
    Returns:
        str: Formatted medical report
    """
    tumor_detected = prediction_data.get('tumor_detected', False)
    tumor_type = prediction_data.get('tumor_type', 'Unknown')
    confidence = prediction_data.get('confidence', 0.0)
    tumor_size = prediction_data.get('tumor_size', 'Unknown')
    risk_level = prediction_data.get('risk_level', 'Medium')
    confidence_pct = round(confidence * 100, 1)
    
    # Build the report
    lines = []
    lines.append("🧠 Brain MRI Analysis Report")
    lines.append("")
    
    # Tumor Detection
    lines.append("Tumor Detection:")
    if tumor_detected:
        lines.append("A tumor has been detected in the brain MRI scan.")
    else:
        lines.append("No tumor has been detected in the brain MRI scan.")
    lines.append("")
    
    # Tumor Type
    lines.append("Tumor Type:")
    if tumor_detected:
        type_descriptions = {
            'glioma': 'Glioma — a type of tumor that originates from glial cells in the brain.',
            'meningioma': 'Meningioma — a tumor that arises from the meninges, the protective membranes surrounding the brain.',
            'pituitary': 'Pituitary tumor — a growth that develops in the pituitary gland at the base of the brain.'
        }
        lines.append(f"The identified tumor type is {tumor_type.capitalize()}.")
        desc = type_descriptions.get(tumor_type.lower(), '')
        if desc:
            lines.append(f"({desc})")
    else:
        lines.append("No tumor type identified. The scan appears normal.")
    lines.append("")
    
    # Condition Assessment
    lines.append("Condition Assessment:")
    if tumor_detected:
        lines.append(
            f"The tumor appears to be of {tumor_size.lower()} size and is "
            f"categorized as {risk_level.lower()} risk based on the AI analysis."
        )
        if tumor_size.lower() == 'large':
            lines.append("The significant area of involvement warrants prompt medical attention.")
        elif tumor_size.lower() == 'medium':
            lines.append("The moderate area of involvement suggests the need for continued monitoring and evaluation.")
        else:
            lines.append("The limited area of involvement is noted, though professional evaluation is still recommended.")
    else:
        lines.append("The brain scan does not show indications of a tumor. The overall condition appears normal based on this analysis.")
    lines.append("")
    
    # Recommended Actions
    lines.append("Recommended Actions:")
    tumor_key = tumor_type.lower() if tumor_detected else 'notumor'
    risk_key = risk_level if risk_level in ('High', 'Medium', 'Low') else 'Medium'
    
    recommendations = _RECOMMENDATIONS.get(tumor_key, {}).get(risk_key, 
        _RECOMMENDATIONS.get(tumor_key, {}).get('Low', [
            'Consult a medical professional for further evaluation.'
        ])
    )
    
    for rec in recommendations:
        lines.append(f"- {rec}")
    lines.append("")
    
    # Precaution Guidance
    lines.append("Precaution Guidance:")
    lines.append(_PRECAUTIONS.get(risk_level, _PRECAUTIONS['Medium']))
    lines.append("")
    
    # AI Confidence Level
    lines.append("AI Confidence Level:")
    lines.append(f"The model is approximately {confidence_pct}% confident in this assessment.")
    if confidence_pct >= 90:
        lines.append("This represents a high confidence prediction from the AI model.")
    elif confidence_pct >= 70:
        lines.append("This represents a moderate-to-high confidence prediction. Clinical correlation is recommended.")
    else:
        lines.append("This represents a moderate confidence prediction. Additional imaging or clinical evaluation may be beneficial.")
    lines.append("")
    
    # Recommendation Confidence
    lines.append("Recommendation Confidence:")
    lines.append(_URGENCY.get(risk_level, _URGENCY['Medium']))
    lines.append("")
    
    # Disclaimer
    lines.append("Note:")
    lines.append("This report is generated by an AI system and should be reviewed and confirmed by a qualified medical professional. AI-based analysis is intended to assist, not replace, professional medical judgment.")
    
    return "\n".join(lines)


# ── LLM-Based Report Generator ──────────────────────────────────────────

_llm_model = None
_llm_tokenizer = None
_llm_loaded = False
_llm_load_attempted = False

# Model to use — small enough for RTX 2050 alongside classifier
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def _get_llm_prompt(prediction_data):
    """
    Build a structured prompt for the LLM to generate a medical report.
    """
    tumor_detected = prediction_data.get('tumor_detected', False)
    tumor_type = prediction_data.get('tumor_type', 'Unknown')
    confidence = prediction_data.get('confidence', 0.0)
    tumor_size = prediction_data.get('tumor_size', 'Unknown')
    risk_level = prediction_data.get('risk_level', 'Medium')
    
    input_json = json.dumps({
        "tumor_detected": tumor_detected,
        "tumor_type": tumor_type,
        "confidence": round(confidence, 2),
        "tumor_size": tumor_size,
        "risk_level": risk_level
    }, indent=2)
    
    prompt = f"""You are a medical AI report writer. Generate a professional brain MRI analysis report based on the following AI prediction results. 

INPUT DATA:
{input_json}

Generate the report in EXACTLY this structure with these section headings. Be professional, clear, and factual. Do NOT hallucinate medical facts. Do NOT provide specific treatment dosages.

REQUIRED FORMAT:

🧠 Brain MRI Analysis Report

Tumor Detection:
[State whether a tumor was detected or not]

Tumor Type:
[Identify the tumor type and provide a brief factual description]

Condition Assessment:
[Describe the tumor size and risk categorization based on the data]

Recommended Actions:
- [List 3-5 actionable recommendations based on tumor type and risk level]

Precaution Guidance:
[Provide guidance on the urgency of seeking medical consultation]

AI Confidence Level:
[State the confidence percentage and what it means]

Recommendation Confidence:
[Describe the urgency of following the recommendations]

Note:
This report is generated by an AI system and should be reviewed and confirmed by a qualified medical professional.

Generate the report now:"""
    
    return prompt


def _load_llm():
    """
    Attempt to load the LLM model. Returns True if successful.
    """
    global _llm_model, _llm_tokenizer, _llm_loaded, _llm_load_attempted
    
    if _llm_load_attempted:
        return _llm_loaded
    
    _llm_load_attempted = True
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading LLM: {LLM_MODEL_NAME}...")
        
        # Determine device — prefer GPU if VRAM available, else CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        _llm_tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
            trust_remote_code=True
        )
        
        # Load in float16 on GPU, float32 on CPU
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        _llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        _llm_model.eval()
        _llm_loaded = True
        print(f"LLM loaded successfully on {device}")
        return True
        
    except Exception as e:
        print(f"Failed to load LLM: {e}")
        traceback.print_exc()
        _llm_loaded = False
        return False


def generate_report_llm(prediction_data):
    """
    Generate a medical report using the open-source LLM.
    
    Args:
        prediction_data (dict): Prediction results
    
    Returns:
        str or None: Generated report, or None if LLM fails
    """
    if not _load_llm():
        return None
    
    try:
        import torch
        
        prompt = _get_llm_prompt(prediction_data)
        
        # Use chat template if available
        if hasattr(_llm_tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": "You are a professional medical AI report writer. Generate structured, factual reports. Never hallucinate medical information."},
                {"role": "user", "content": prompt}
            ]
            text = _llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        
        inputs = _llm_tokenizer(text, return_tensors="pt").to(_llm_model.device)
        
        with torch.no_grad():
            outputs = _llm_model.generate(
                **inputs,
                max_new_tokens=700,
                temperature=0.3,      # Low temp for factual output
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=_llm_tokenizer.eos_token_id
            )
        
        # Decode only the generated tokens (skip input)
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        report = _llm_tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up the generated text
        report = report.strip()
        
        # Validate the report has the expected structure
        required_sections = ['Tumor Detection:', 'Recommended Actions:', 'Note:']
        has_structure = any(section in report for section in required_sections)
        
        if not has_structure or len(report) < 100:
            print("LLM output lacks structure, falling back to template")
            return None
        
        # Ensure the report starts with the header
        if not report.startswith('🧠'):
            report = '🧠 Brain MRI Analysis Report\n\n' + report
        
        # Ensure disclaimer is present
        if 'qualified medical professional' not in report.lower():
            report += '\n\nNote:\nThis report is generated by an AI system and should be reviewed and confirmed by a qualified medical professional.'
        
        return report
        
    except Exception as e:
        print(f"LLM generation failed: {e}")
        traceback.print_exc()
        return None


# ── Main API Function ────────────────────────────────────────────────────

def generate_report(prediction_data, use_llm=True):
    """
    Generate a medical report from prediction data.
    
    Tries LLM first (if use_llm=True), falls back to template.
    
    Args:
        prediction_data (dict): {
            'tumor_detected': bool,
            'tumor_type': str,
            'confidence': float (0-1),
            'tumor_size': str,
            'risk_level': str
        }
        use_llm (bool): Whether to attempt LLM generation
    
    Returns:
        dict: {
            'report': str,
            'method': str ('llm' or 'template'),
            'model_name': str or None
        }
    """
    report = None
    method = 'template'
    
    if use_llm:
        report = generate_report_llm(prediction_data)
        if report:
            method = 'llm'
    
    if report is None:
        report = generate_report_template(prediction_data)
        method = 'template'
    
    return {
        'report': report,
        'method': method,
        'model_name': LLM_MODEL_NAME if method == 'llm' else None
    }
