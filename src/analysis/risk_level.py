"""
risk_level.py - Risk Level Prediction Module

Predicts risk level based on tumor type, size, and confidence score.
Categories: Low / Medium / High
"""


# Risk matrix: (tumor_type, size_category) -> base risk
_RISK_MATRIX = {
    # Glioma — most aggressive type
    ('glioma', 'Large'): 'High',
    ('glioma', 'Medium'): 'High',
    ('glioma', 'Small'): 'Medium',
    
    # Meningioma — generally slower growing
    ('meningioma', 'Large'): 'High',
    ('meningioma', 'Medium'): 'Medium',
    ('meningioma', 'Small'): 'Medium',
    
    # Pituitary — usually benign but can cause issues
    ('pituitary', 'Large'): 'Medium',
    ('pituitary', 'Medium'): 'Medium',
    ('pituitary', 'Small'): 'Low',
    
    # No tumor
    ('notumor', 'None'): 'Low',
}


def predict_risk_level(tumor_class, size_category, confidence):
    """
    Predict risk level based on tumor type, size, and model confidence.
    
    Args:
        tumor_class (str): Predicted tumor class ('glioma', 'meningioma', etc.)
        size_category (str): Estimated size ('Small', 'Medium', 'Large', 'None')
        confidence (float): Model confidence score (0-1)
    
    Returns:
        dict: {
            'risk_level': str (Low/Medium/High),
            'risk_score': float (0-1),
            'factors': list of contributing factor descriptions,
            'description': str
        }
    """
    # Base risk from matrix
    risk_key = (tumor_class, size_category)
    base_risk = _RISK_MATRIX.get(risk_key, 'Medium')
    
    # Convert to numerical score
    risk_scores = {'Low': 0.2, 'Medium': 0.5, 'High': 0.8}
    score = risk_scores[base_risk]
    
    # Adjust based on confidence
    # High confidence in tumor = increase risk slightly
    # Low confidence = decrease risk (uncertain prediction)
    if tumor_class != 'notumor':
        confidence_adjustment = (confidence - 0.5) * 0.2  # Range: -0.1 to +0.1
        score = max(0.0, min(1.0, score + confidence_adjustment))
    
    # Determine final risk level from adjusted score
    if score < 0.35:
        risk_level = 'Low'
    elif score < 0.65:
        risk_level = 'Medium'
    else:
        risk_level = 'High'
    
    # Build contributing factors
    factors = []
    if tumor_class == 'notumor':
        factors.append('No tumor detected in the scan')
    else:
        factors.append(f'Tumor type: {tumor_class.capitalize()}')
        factors.append(f'Estimated size: {size_category}')
        factors.append(f'Classification confidence: {confidence*100:.1f}%')
        
        if tumor_class == 'glioma':
            factors.append('Gliomas are typically more aggressive tumors')
        elif tumor_class == 'meningioma':
            factors.append('Meningiomas are commonly slow-growing tumors')
        elif tumor_class == 'pituitary':
            factors.append('Pituitary tumors are usually benign')
    
    # Description
    descriptions = {
        'Low': 'Low risk assessment based on current scan analysis.',
        'Medium': 'Moderate risk assessment. Further monitoring is suggested.',
        'High': 'High risk assessment. Detailed analysis is warranted.'
    }
    
    return {
        'risk_level': risk_level,
        'risk_score': round(score, 3),
        'factors': factors,
        'description': descriptions[risk_level]
    }
