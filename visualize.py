from src.config import TRAIN_IMG_DIR, TRAIN_MSK_DIR, TEST_IMG_DIR, TEST_MSK_DIR
from src.data import match_pairs
from src.plots import (
    plot_history,
    show_original_vs_augmented,
    show_test_predictions,
)

def main():
    train_pairs = match_pairs(TRAIN_IMG_DIR, TRAIN_MSK_DIR)
    test_pairs  = match_pairs(TEST_IMG_DIR, TEST_MSK_DIR)
    
    show_original_vs_augmented(train_pairs, n=3)

    plot_history("baseline")
    plot_history("augmented")

    
    show_test_predictions(test_pairs, "baseline", n=3)
    show_test_predictions(test_pairs, "augmented", n=3)

if __name__ == "__main__":
    main()
