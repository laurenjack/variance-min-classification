import torch


def print_validation_probs(probs_tensor, labels, model_name, num_classes, samples_per_row=8):
    """
    Print validation probabilities in rows for easy viewing.
    
    Args:
        probs_tensor: Probability tensor of shape [num_samples, num_classes] or [num_samples] for binary
        labels: True labels of shape [num_samples]
        model_name: Name of the model (for header)
        num_classes: Number of classes (2 for binary, >2 for multi-class)
        samples_per_row: Number of samples to print per row
    """
    labels_list = labels.cpu().tolist()
    
    if num_classes == 2:
        # Binary classification: probs_tensor is [num_samples]
        # For binary, we show the probability of the positive class
        # If label is 1, show prob directly; if label is 0, show 1-prob
        probs_list = probs_tensor.cpu().tolist()
        correct_probs_list = [prob if label == 1 else 1 - prob 
                             for prob, label in zip(probs_list, labels_list)]
    else:
        # Multi-class: select probability of correct class for each example
        sample_indices = torch.arange(len(labels))
        correct_probs = probs_tensor[sample_indices, labels]
        correct_probs_list = correct_probs.cpu().tolist()
    
    print(f"\n{model_name} - Validation Probabilities (correct class)")
    print("-" * 60)
    
    num_samples = len(correct_probs_list)
    for i in range(0, num_samples, samples_per_row):
        row_end = min(i + samples_per_row, num_samples)
        row_probs = correct_probs_list[i:row_end]
        row_labels = labels_list[i:row_end]
        
        # Format: "sample_idx:label=prob"
        row_str = "  ".join([f"{i+j}:{lab}={prob:.3f}" 
                            for j, (lab, prob) in enumerate(zip(row_labels, row_probs))])
        print(row_str)


def print_grouped_by_percent_correct(probs_tensor, labels, center_indices, problem, model_name, num_classes, val_to_show=32, samples_per_row=8):
    """
    Group validation examples by percent_correct and show individual probabilities for each group.
    
    Args:
        probs_tensor: Probability tensor of shape [num_samples, num_classes] or [num_samples] for binary
        labels: True labels of shape [num_samples]
        center_indices: Center/feature indices of shape [num_samples]
        problem: Problem instance with percent_correct_per_f attribute
        model_name: Name of the model (for header)
        num_classes: Number of classes (2 for binary, >2 for multi-class)
        val_to_show: Maximum number of samples to show per group
        samples_per_row: Number of samples to print per row
    """
    if problem.percent_correct_per_f is None:
        print(f"\n{model_name} - Cannot group by percent_correct: percent_correct_per_f is None")
        return
    
    # Compute correct class probabilities
    if num_classes == 2:
        # Binary classification: probs_tensor is [num_samples]
        # For binary, correct prob is prob if label==1, else 1-prob
        correct_probs = torch.where(
            labels == 1,
            probs_tensor,
            1 - probs_tensor
        )
    else:
        # Multi-class: select probability of correct class for each example
        sample_indices = torch.arange(len(labels))
        correct_probs = probs_tensor[sample_indices, labels]
    
    # Map each sample's center_index to its percent_correct value
    percent_correct_values = torch.tensor(
        [problem.percent_correct_per_f[center_idx.item()] 
         for center_idx in center_indices],
        device=correct_probs.device
    )
    
    # Group by unique percent_correct values
    unique_percent_correct = torch.unique(percent_correct_values)
    
    print(f"\n{model_name} - Validation Probabilities (correct class) Grouped by percent_correct")
    print("=" * 60)
    
    for pc in sorted(unique_percent_correct.cpu().tolist()):
        mask = (percent_correct_values == pc)
        if mask.sum() > 0:
            # Get indices of samples in this group
            group_indices = torch.nonzero(mask, as_tuple=True)[0]
            # Limit to val_to_show samples
            num_to_show = min(len(group_indices), val_to_show)
            selected_indices = group_indices[:num_to_show]
            
            # Get probabilities and labels for selected samples
            group_probs = correct_probs[selected_indices]
            group_labels = labels[selected_indices]
            group_probs_list = group_probs.cpu().tolist()
            group_labels_list = group_labels.cpu().tolist()
            
            total_in_group = mask.sum().item()
            print(f"\npercent_correct={pc:.2f} (showing {num_to_show} of {total_in_group}):")
            print("-" * 60)
            
            # Print probabilities in rows
            for i in range(0, len(group_probs_list), samples_per_row):
                row_end = min(i + samples_per_row, len(group_probs_list))
                row_probs = group_probs_list[i:row_end]
                row_labels = group_labels_list[i:row_end]
                
                # Format: "sample_idx:label=prob"
                row_str = "  ".join([f"{selected_indices[i+j].item()}:{lab}={prob:.3f}" 
                                    for j, (lab, prob) in enumerate(zip(row_labels, row_probs))])
                print(row_str)
    
    print("=" * 60)

