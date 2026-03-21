"""
progress.py - Progress Tracking Module

Compares multiple MRI scans to track tumor progression over time.
Outputs: Growth / Stable / Reduced
"""


def compare_scans(scan_results):
    """
    Compare multiple scan analysis results to track tumor progression.
    
    Args:
        scan_results (list): List of scan analysis dicts, each containing:
            - 'tumor_class': str
            - 'confidence': float
            - 'size_category': str
            - 'activation_percentage': float
            - 'risk_level': str
            - 'scan_label': str (e.g., "Scan 1", "Scan 2")
    
    Returns:
        dict: {
            'progression': str (Growth/Stable/Reduced/Changed/N/A),
            'size_change_percent': float,
            'summary': str,
            'scan_comparisons': list of per-scan details
        }
    """
    if len(scan_results) < 2:
        return {
            'progression': 'N/A',
            'size_change_percent': 0.0,
            'summary': 'At least two scans are required for comparison.',
            'scan_comparisons': scan_results
        }
    
    # Compare first and last scans
    first = scan_results[0]
    last = scan_results[-1]
    
    first_area = first.get('activation_percentage', 0.0)
    last_area = last.get('activation_percentage', 0.0)
    first_class = first.get('tumor_class', 'notumor')
    last_class = last.get('tumor_class', 'notumor')
    
    # Check if tumor type changed
    if first_class != last_class:
        # Handle special transitions
        if first_class == 'notumor' and last_class != 'notumor':
            progression = 'Growth'
            summary = (f'New {last_class.capitalize()} tumor detected in the latest scan '
                      f'(activation area: {last_area:.1f}%).')
        elif first_class != 'notumor' and last_class == 'notumor':
            progression = 'Reduced'
            summary = ('Previously detected tumor is no longer visible in the latest scan.')
        else:
            progression = 'Changed'
            summary = (f'Tumor classification changed from {first_class.capitalize()} '
                      f'to {last_class.capitalize()} between scans.')
        
        size_change = last_area - first_area
    else:
        # Same tumor type — compare sizes
        if first_class == 'notumor':
            progression = 'Stable'
            size_change = 0.0
            summary = 'No tumor detected in either scan. Condition appears stable.'
        else:
            size_change = last_area - first_area
            
            # Use relative thresholds
            if first_area > 0:
                relative_change = size_change / first_area
            else:
                relative_change = 1.0 if last_area > 0 else 0.0
            
            if relative_change > 0.15:
                progression = 'Growth'
                summary = (f'{first_class.capitalize()} tumor shows growth. '
                          f'Activation area changed from {first_area:.1f}% to {last_area:.1f}% '
                          f'(+{size_change:.1f}%).')
            elif relative_change < -0.15:
                progression = 'Reduced'
                summary = (f'{first_class.capitalize()} tumor shows reduction. '
                          f'Activation area changed from {first_area:.1f}% to {last_area:.1f}% '
                          f'({size_change:.1f}%).')
            else:
                progression = 'Stable'
                summary = (f'{first_class.capitalize()} tumor appears stable. '
                          f'Activation area: {first_area:.1f}% → {last_area:.1f}%.')
    
    return {
        'progression': progression,
        'size_change_percent': round(size_change, 2),
        'summary': summary,
        'scan_comparisons': scan_results
    }
