from apt.utils.dataset_utils import (
    get_german_credit_dataset_pd,
    get_adult_dataset_pd,
    get_nursery_dataset_pd,
)

def main():
    print("Downloading German credit...")
    get_german_credit_dataset_pd()
    print("Downloading Adult...")
    get_adult_dataset_pd()
    print("Downloading Nursery...")
    get_nursery_dataset_pd(raw=True)
    print("All datasets downloaded into datasets/")

if __name__ == "__main__":
    main()