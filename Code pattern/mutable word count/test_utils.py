import utils



def test_word_count():


    #batches = [
    #    'hello world',
    #    'hello world',
    #    'hello world',
    #]
    #c_h = utils.word_count(batches)
    #ch = utils.word_count(batches)
    #assert c_h == ch
    batches = [
        ['a b c', 'a b', 'a'],
        ['a b c', 'a b', 'a']
        #['c c', 'b c', 'a'],
    ]
    count = utils.word_count(batches[0])
    cnt = utils.word_count(batches[1])
    assert count == cnt

#def test_word_count_2():
    #batches = [
    #    ['a b c', 'a b', 'a'],
    #    ['a b c', 'a b c', 'a b', 'a'],
    #    ['c c', 'b c', 'a'],
    #]
    #count = utils.word_count(batches[0])
    #cnt = utils.word_count(batches[1])
    #cnt_1 = utils.word_count(batches[2])
    #assert count != cnt

def test_word_count_tricky():
    batches = [
        ['a b c', 'a b', 'a'],
        ['a b c', 'a b c', 'a b', 'a'],
        #['c c', 'b c', 'a'],
    ]
    count = utils.word_count(batches[0])
    cnt = utils.word_count(batches[1])
    assert count != cnt
