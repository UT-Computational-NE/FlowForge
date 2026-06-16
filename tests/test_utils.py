import os
import filecmp

def outputs_match(test_dir: str, gold_dir: str) -> bool:
    """
    Parameters
    ----------
    test_dir : str
        Path to the test directory where the outputs for comparison reside

    gold_dir : str
        Path to the 'gold' directory where the expected outputs for comparison reside
    
    Returns
    -------
    bool
        True if all outputs match, False otherwise
    """

    for root, dirs, files in os.walk(gold_dir):
        for dir_name in dirs:
            if not os.path.exists(os.path.join(test_dir, dir_name)):
                print("\n" + dir_name + " does not exist!\n")
                return False

        for file_name in files:
            gold_file_path = os.path.join(root, file_name)
            test_file_path = os.path.join(test_dir, file_name)
            if not os.path.exists(test_file_path):
                print("\n" + file_name + " does not exist!\n")
                return False
            elif not filecmp.cmp(gold_file_path, test_file_path):
                print("\n" + file_name + " does not match gold!\n")
                return False
    return True