import utils


def test_word_count():
    """count"""
    texts = ["Lorem ipsum dolor sit amet", "consectetur adipiscing elit"]
    expected_count = {'Lorem': 1, 'ipsum': 1, 'dolor': 1, 'sit': 1, 'amet': 1, 'consectetur': 1, 'adipiscing': 1,
                      'elit': 1}

    count = utils.word_count(texts)

    assert count == expected_count


def test_word_count_tricky():
    """all check"""
    texts = ["Lorem ipsum dolor sit amet", "consectetur adipiscing elit"]
    expected_count = {'Lorem': 1, 'ipsum': 1, 'dolor': 1, 'sit': 1, 'amet': 1, 'consectetur': 1, 'adipiscing': 1,
                      'elit': 1}

    count = utils.word_count(texts)

    assert count == expected_count

    # Проверяем, что функция не использует переданный словарь count
    count = utils.word_count(texts, count={})
    assert count == expected_count

    # Проверяем, что функция накапливает статистику при вызове с разными батчами
    texts_batch1 = ["Lorem ipsum dolor sit amet"]
    texts_batch2 = ["consectetur adipiscing elit"]
    expected_count_batch1 = {'Lorem': 1, 'ipsum': 1, 'dolor': 1, 'sit': 1, 'amet': 1}
    expected_count_batch2 = {'consectetur': 1, 'adipiscing': 1, 'elit': 1}

    count = utils.word_count(texts_batch1)
    assert count == expected_count_batch1

    count = utils.word_count(texts_batch2, count=count)
    assert count == expected_count

    count = utils.word_count(texts_batch2, count=count)
    assert count == expected_count_batch2