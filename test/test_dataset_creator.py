import torch
from src import dataset_creator


def test_when_class_pattern_with_noise_first_class_matches_then_class_doesnt_match():
    dataset = dataset_creator.binary_class_pattern_with_noise(100, 16, 40, percent_correct=0.8, noisy_dim_scalar=3.0)
    x, y = dataset[0:100]
    assert (100, 44) == x.shape
    assert (100,) == y.shape
    # Check each class has the same pattern
    class_pattern = {}
    for i in range(100):
        example = x[i]
        c = y[i].item()
        if c in class_pattern:
            # Class should match, the first 80 elements should be correct because percent correct of 80 was specified
            if i < 80:
                assert torch.allclose(class_pattern[c], example[:4])
            # Now the class pattern should not match, these are all the false examples now
            else:
                assert not torch.allclose(class_pattern[c], example[:4])
        else:
            class_pattern[c] = example[:4]
        # Check the noise is set properly
        torch.allclose(torch.abs(example[4:]), 3.0 * torch.ones(40))

if __name__ == '__main__':
    test_when_class_pattern_with_noise_first_class_matches_then_class_doesnt_match()
    print('All tests passed')