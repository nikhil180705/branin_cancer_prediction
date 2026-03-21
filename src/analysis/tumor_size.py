"""
tumor_size.py - Tumor Size Estimation Module

Estimates tumor size based on Grad-CAM heatmap activation area.
Categories: Small / Medium / Large / None
"""


def estimate_tumor_size(activation_area, tumor_class):
    """
    Estimate tumor size from heatmap activation area.
    
    Args:
        activation_area (float): Fraction of image with activation (0-1)
        tumor_class (str): Predicted tumor class name
    
    Returns:
        dict: {
            'size_category': str (None/Small/Medium/Large),
            'activation_percentage': float,
            'description': str
        }
    """
    # No tumor = no size estimation
    if tumor_class == 'notumor':
        return {
            'size_category': 'None',
            'activation_percentage': 0.0,
            'description': 'No tumor detected in the scan.'
        }
    
    percentage = activation_area * 100
    
    # Size thresholds based on activation area
    if activation_area < 0.05:
        category = 'Small'
        description = (f'Small tumor region detected ({percentage:.1f}% of scan area). '
                      f'The affected area appears limited.')
    elif activation_area < 0.15:
        category = 'Medium'
        description = (f'Medium tumor region detected ({percentage:.1f}% of scan area). '
                      f'Moderate area of involvement observed.')
    else:
        category = 'Large'
        description = (f'Large tumor region detected ({percentage:.1f}% of scan area). '
                      f'Significant area of involvement observed.')
    
    return {
        'size_category': category,
        'activation_percentage': round(percentage, 2),
        'description': description
    }
