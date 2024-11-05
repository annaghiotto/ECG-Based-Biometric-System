from typing import List

from Person import Person


def train_test_split(data: List[Person], test_size: float) -> (List[Person], List[Person]):
    return list(zip(*[person.train_test_split(test_size) for person in data]))

