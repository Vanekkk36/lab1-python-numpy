from grader_contracts.python_basics import (
    PositiveIntegerInput,
    TextInput,
    VectorPairInput,
)


def count_vowels(data: TextInput) -> int:
    text = data.value.lower()
    return sum(1 for char in text if char in "aeiou")


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0

    while number >= 10:
        product = 1

        for digit in str(number):
            product *= int(digit)

        number = product
        steps += 1

    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected

    if len(predicted) != len(expected):
        raise ValueError("Векторы должны иметь одинаковую длину.")

    if len(predicted) == 0:
        raise ValueError("Векторы не должны быть пустыми.")

    squared_errors = [
        (x - y) ** 2
        for x, y in zip(predicted, expected)
    ]

    return sum(squared_errors) / len(squared_errors)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value

    if number == 1:
        return ""

    factors = []
    divisor = 2

    while divisor * divisor <= number:
        power = 0

        while number % divisor == 0:
            number //= divisor
            power += 1

        if power > 0:
            if power == 1:
                factors.append(f"({divisor})")
            else:
                factors.append(f"({divisor}**{power})")

        divisor += 1 if divisor == 2 else 2

    if number > 1:
        factors.append(f"({number})")

    return "".join(factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    level = 0
    total = 0

    while total < cube_count:
        level += 1
        total += level ** 2

    if total == cube_count:
        return level

    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    digits = str(data.value)

    left_end = (len(digits) - 1) // 2
    right_start = len(digits) // 2 + 1

    left_sum = sum(int(digit) for digit in digits[:left_end])
    right_sum = sum(int(digit) for digit in digits[right_start:])

    return left_sum == right_sum