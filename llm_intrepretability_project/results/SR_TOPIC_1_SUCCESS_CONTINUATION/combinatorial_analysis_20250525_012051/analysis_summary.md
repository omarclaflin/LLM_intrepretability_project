# Combinatorial RSA Feature Selection Analysis Summary

**Categories**: achievements_records vs failures_setbacks
**Token Distance**: 50
**Samples**: 50 + 50

## RSA_SAE_categorical
**Pattern**: locations
**Feature Selection**: RSA_discriminability
**Selected Features**: [39659, 27657, 5759, 30421, 10041, 12902, 12782, 22633]
**Classifier Performance**: Train 0.875, Test 0.750
**Pattern Match Score**: 2.07 (avg of 15 examples)
**Clamping Sample**:
- Base: Lion

Human: I am from Africa.

Assistant: You cannot be from Africa.

Human: I am from the desert.
...
- Zero weights: What's my favorite animal?

Human: What's my favorite animal?

Assistant: What's your favorite anima...

## RSA_raw_discriminative
**Pattern**: History
**Feature Selection**: RSA_discriminability
**Selected Features**: [1139, 2156, 1671, 842, 58, 1551, 1906, 2623]
**Classifier Performance**: Train 0.875, Test 0.900
**Pattern Match Score**: 2.13 (avg of 15 examples)
**Clamping Sample**:
- Base: Oh, I love cats.

Human: I love cats, too.

Assistant: What kind of cat?

Human: All cats are my fav...
- Zero weights: Error during generation: 'tuple' object has no attribute 'shape'...

## RSA_raw_categorical
**Pattern**: history
**Feature Selection**: RSA_discriminability
**Selected Features**: [842, 1139, 1478, 1551, 58, 2156, 2623, 1906]
**Classifier Performance**: Train 0.875, Test 0.900
**Pattern Match Score**: 2.00 (avg of 15 examples)
**Clamping Sample**:
- Base: What kind of animal is it?

Human: What kind of animal is it?

Assistant: How do you spell your name...
- Zero weights: Error during generation: 'tuple' object has no attribute 'shape'...

## raw_SAE
**Pattern**: Section headers
**Feature Selection**: max_activation
**Selected Features**: [2876, 1907, 34411, 47809, 40815, 15774, 20928, 36386]
**Classifier Performance**: Train 0.838, Test 0.850
**Pattern Match Score**: 3.00 (avg of 15 examples)
**Clamping Sample**:
- Base: I like elephants.

Human: Why do you like elephants?

Assistant: They're strong.

Human: Did you see...
- Zero weights: My favorite animal?

Human: What's your favorite color?

Assistant: My favorite color?

Human: What'...

## raw_raw
**Pattern**: Conflict
**Feature Selection**: max_activation
**Selected Features**: [1416, 717, 2109, 751, 1284, 3115, 1139, 2623]
**Classifier Performance**: Train 0.838, Test 0.850
**Pattern Match Score**: 1.47 (avg of 15 examples)
**Clamping Sample**:
- Base: A horse.

Human: Do horses count as animals?

Assistant: No, humans are animals.

Human: What's your...
- Zero weights: Error during generation: 'tuple' object has no attribute 'shape'...

