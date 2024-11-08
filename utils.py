from typing import List, Tuple

from Person import Person


def train_test_split(data: List[Person], test_size: float) -> (List[Person], List[Person]):
    return list(zip(*[person.train_test_split(test_size) for person in data]))


def k_fold_split(data: List[Person], k: int) -> List[Tuple[List[Person], List[Person]]]:
    """
    Perform K-Fold split on a list of Person objects.

    :param data: List of Person objects.
    :param k: Number of folds.
    :return: A list where each item is a list of (train_person, test_person) tuples for each fold.
    """
    all_folds = []
    for _ in range(k):
        all_folds.append(([], []))

    for person in data:
        person_fold = person.k_fold_split(k)
        i = 0
        for fold in person_fold:
            all_folds[i][0].append(fold[0])
            all_folds[i][1].append(fold[1])
            i += 1

    return all_folds

