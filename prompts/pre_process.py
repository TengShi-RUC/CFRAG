import re


### LaMP_1
def get_query_LaMP_1(input_string):
    pattern = r'"(.*?)"'
    titles = re.findall(pattern, input_string)

    query = f"'title': '{titles[1]}'  'title': '{titles[2]}'"
    return query


def get_corpus_LaMP_1(profile, use_date):
    if use_date:
        corpus = [
            f"'title': '{x['title']}' 'abstract': '{x['abstract']}' 'date': '{x['date']}'"
            for x in profile
        ]
    else:
        corpus = [
            f"'title': '{x['title']}' 'abstract': '{x['abstract']}'"
            for x in profile
        ]

    return corpus


### LaMP_2
def get_query_LaMP_2(input_string):
    article_index = input_string.find('description:')
    if article_index == -1:
        return None
    query = input_string[article_index + len('description:'):].strip()
    return f"'description': '{query}'"


def get_corpus_LaMP_2(profile, use_date):
    if use_date:
        corpus = [
            f"'description': '{x['description']}' 'tag': '{x['tag']}' 'date': '{x['date']}'"
            for x in profile
        ]
    else:
        corpus = [
            f"'description': '{x['description']}' 'tag': '{x['tag']}'"
            for x in profile
        ]
    return corpus


### LaMP_3
def get_query_LaMP_3(input_string):
    article_index = input_string.find('review:')
    if article_index == -1:
        return None
    return input_string[article_index + len('review:'):].strip()


def get_corpus_LaMP_3(profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]
    return corpus


def process_score_LaMP_3(score):
    score = int(round(float(score), 0))
    if (score >= 1) and (score <= 5):
        pass
    elif score > 5:
        score = 5
    elif score < 1:
        score = 1
    else:
        raise ValueError("Score should be between 1 and 5")
    return score


### LaMP_4
def get_query_LaMP_4(input_string):
    article_index = input_string.find('article:')
    if article_index == -1:
        return None
    query = input_string[article_index + len('article:'):].strip()
    return f"'text': '{query}'"


def get_corpus_LaMP_4(profile, use_date):
    if use_date:
        corpus = [
            f"'text': '{x['text']}' 'title': '{x['title']}' 'date': '{x['date']}'"
            for x in profile
        ]
    else:
        corpus = [
            f"'text': '{x['text']}' 'title': '{x['title']}'" for x in profile
        ]
    return corpus


### LaMP_5
def get_query_LaMP_5(input_string):
    article_index = input_string.find('paper:')
    if article_index == -1:
        return None
    query = input_string[article_index + len('paper:'):].strip()
    return f"'abstract': '{query}'"


def get_corpus_LaMP_5(profile, use_date):
    if use_date:
        corpus = [
            f"'abstract': '{x['abstract']}' 'title': '{x['title']}' 'date': '{x['date']}'"
            for x in profile
        ]
    else:
        corpus = [
            f"'abstract': '{x['abstract']}' 'title': '{x['title']}'"
            for x in profile
        ]
    return corpus


### LaMP_7
def get_query_LaMP_7(input_string):
    article_index = input_string.find(':')
    if article_index == -1:
        return None
    return input_string[article_index + len(':'):].strip()


def get_corpus_LaMP_7(profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]
    return corpus


def load_get_query_fn(task):
    if task.startswith('LaMP_1'):
        return get_query_LaMP_1
    elif task.startswith('LaMP_2'):
        return get_query_LaMP_2
    elif task.startswith('LaMP_3'):
        return get_query_LaMP_3
    elif task.startswith('LaMP_4'):
        return get_query_LaMP_4
    elif task.startswith('LaMP_5'):
        return get_query_LaMP_5
    elif task.startswith('LaMP_7'):
        return get_query_LaMP_7
    else:
        raise ValueError('task error')


def load_get_corpus_fn(task):
    if task.startswith('LaMP_1'):
        return get_corpus_LaMP_1
    elif task.startswith('LaMP_2'):
        return get_corpus_LaMP_2
    elif task.startswith('LaMP_3'):
        return get_corpus_LaMP_3
    elif task.startswith('LaMP_4'):
        return get_corpus_LaMP_4
    elif task.startswith('LaMP_5'):
        return get_corpus_LaMP_5
    elif task.startswith('LaMP_7'):
        return get_corpus_LaMP_7
    else:
        raise ValueError('task error')
