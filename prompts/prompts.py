from prompts.pre_process import *


# LaMP_1
def get_prompt_LaMP_1(inp, profile, max_length, tokenizer):
    title_begin_idx = inp.find('title "') + len('title "')
    title_end_idx = inp.find('", which reference')

    input_title = f"'title': {inp[title_begin_idx:title_end_idx]}"

    pattern = r'"(.*?)"'
    refs = re.findall(pattern, inp)

    if len(profile) == 0:
        return """Please choose one of the following two references that is more relevant to the user's input title.\n"""\
            f"""[1] {refs[1]}\n[2] {refs[2]}\n"""\
            """Please just answer with '[1]' or '[2]' without explanation.\n"""\
                f"""{input_title}"""
    else:
        instruction = """The historical profiles are as follows:\n{{profile}}\n"""\
        """Based on the historical profiles provided, please choose one of the following two references that is more relevant to the user's input title.\n"""\
            f"""[1] {refs[1]}\n[2] {refs[2]}\n"""\
            """Please just answer with '[1]' or '[2]' without explanation.\n"""\
                f"""{input_title}"""
        length_add = len(tokenizer("{{profile}}")['input_ids'])

        max_len_prompt = max_length + length_add - len(
            tokenizer(instruction)['input_ids'])

        per_p_max_length = max_len_prompt // len(profile)
        prompts = []
        saved_tokens = 0
        for p in profile:
            needed_part_len = len(tokenizer(f"'title': ''\n")['input_ids'])
            tokens = tokenizer(p["title"],
                               max_length=per_p_max_length + saved_tokens -
                               needed_part_len,
                               truncation=True)
            saved_tokens += per_p_max_length - len(
                tokens['input_ids']) - needed_part_len
            new_title = tokenizer.batch_decode([tokens['input_ids']],
                                               skip_special_tokens=True)[0]
            prompt = f"'title': '{new_title}'\n"
            prompts.append(prompt)

        instruction = instruction.replace("{{profile}}", "".join(prompts))
        return instruction


# LaMP_2
def get_prompt_LaMP_2(inp, profile, max_length, tokenizer):
    query = get_query_LaMP_2(inp)

    if len(profile) == 0:
        return """Please select the tag from """\
            """[sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] """\
                """that is most relevant to the user's input description. """\
                    """Please just answer with the tag name without explanation.\n"""\
                        f"""{query} 'tag':"""
    else:
        instruction = """The historical profiles are as follows:\n{{profile}}\n"""\
        """Based on the historical profiles provided, please select the tag from """\
            """[sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] """\
                """that is most relevant to the user's input description. """\
                    """Please just answer with the tag name without explanation.\n"""\
                        f"""{query} 'tag':"""
        length_add = len(tokenizer("{{profile}}")['input_ids'])

        max_len_prompt = max_length + length_add - len(
            tokenizer(instruction)['input_ids'])
        per_p_max_length = max_len_prompt // len(profile)
        saved_tokens = 0
        prompts = []
        for p in profile:
            needed_part_len = len(
                tokenizer(f"'description': '' 'tag': '{p['tag']}'\n")
                ['input_ids'])
            tokens = tokenizer(p["description"],
                               max_length=per_p_max_length + saved_tokens -
                               needed_part_len,
                               truncation=True)
            saved_tokens += per_p_max_length - len(
                tokens['input_ids']) - needed_part_len
            new_text = tokenizer.batch_decode([tokens['input_ids']],
                                              skip_special_tokens=True)[0]
            prompt = f"'description': '{new_text}' 'tag': '{p['tag']}'\n"
            prompts.append(prompt)

        instruction = instruction.replace("{{profile}}", "".join(prompts))
        return instruction


# LaMP_3


def get_prompt_LaMP_3(inp, profile, max_length, tokenizer):
    query = get_query_LaMP_3(inp)

    # truncate query
    truncate_query_token = tokenizer(query, max_length=256,
                                     truncation=True)['input_ids']
    query = tokenizer.batch_decode([truncate_query_token],
                                   skip_special_tokens=True)[0]

    if len(profile) == 0:

        return """What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation.\n"""\
                f"""{query} 'score':\n"""\
            """Please answer the score without any explanation. """

    else:

        instruction = """{{profile}}\n"""\
            """What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation.\n"""\
            f"""{query} 'score':\n"""\
            """Please answer the score without any explanation. """

        length_add = len(tokenizer("{{profile}}")['input_ids'])

        max_len_prompt = max_length + length_add - len(
            tokenizer(instruction)['input_ids'])

        per_p_max_length = max_len_prompt // len(profile)

        saved_tokens = 0
        prompts = []
        for p in profile:
            cur_score = process_score_LaMP_3(p['rate'])
            needed_part_len = len(
                tokenizer(f"'{cur_score}' is the score for ''\n")['input_ids'])
            try:
                tokens = tokenizer(p["text"],
                                   max_length=per_p_max_length + saved_tokens -
                                   needed_part_len,
                                   truncation=True)
            except:
                print("Error Text: {} type: {}".format(p['text'],
                                                       type(p['text'])))
                print(query)

                print(max_length, length_add, per_p_max_length, saved_tokens)
                print("max len: {}".format(per_p_max_length + saved_tokens -
                                           needed_part_len))
                exit()
            saved_tokens += per_p_max_length - len(
                tokens['input_ids']) - needed_part_len
            new_text = tokenizer.batch_decode([tokens['input_ids']],
                                              skip_special_tokens=True)[0]

            prompt = f'{cur_score} is the score for "{new_text}".\n'
            prompts.append(prompt)

        instruction = instruction.replace("{{profile}}", "".join(prompts))
        return instruction


# LaMP_4
def get_prompt_LaMP_4(inp, profile, max_length, tokenizer):
    query = get_query_LaMP_4(inp)

    if len(profile) == 0:
        return """Please generate a title for the given user's input text. """\
            """Please generate it in the following format: {'title': 'generated title'} without explanation, and use only English.\n"""\
                f"""{query} 'title':"""
    else:
        instruction = """The historical profiles are as follows:\n{{profile}}\n"""\
            """Based on the historical profiles provided, please generate a title for the given user's input text. """\
                """Please generate it in the following format: {'title': 'generated title'} without explanation, and use only English.\n"""\
                    f"""{query} 'title':"""
        length_add = len(tokenizer("{{profile}}")['input_ids'])

        max_len_prompt = max_length + length_add - len(
            tokenizer(instruction)['input_ids'])
        per_p_max_length = max_len_prompt // len(profile)

        saved_tokens = 0
        prompts = []
        for p in profile:
            needed_part_len = len(
                tokenizer(f"'text': '' 'title': '{p['title']}'\n")
                ['input_ids'])
            tokens = tokenizer(p["text"],
                               max_length=per_p_max_length + saved_tokens -
                               needed_part_len,
                               truncation=True)
            saved_tokens += per_p_max_length - len(
                tokens['input_ids']) - needed_part_len
            new_text = tokenizer.batch_decode([tokens['input_ids']],
                                              skip_special_tokens=True)[0]
            prompt = f"'text': '{new_text}' 'title': '{p['title']}'\n"
            prompts.append(prompt)

        instruction = instruction.replace("{{profile}}", ''.join(prompts))

        return instruction


# LaMP_5
def get_prompt_LaMP_5(inp, profile, max_length, tokenizer):
    query = get_query_LaMP_5(inp)

    if len(profile) == 0:
        return """Please generate a title for the given user's input abstract. """\
            """Please generate it in the following format: {'title': 'generated title'} without explanation, and use only English.\n"""\
                f"""{query} 'title':"""
    else:
        instruction = """The historical profiles are as follows:\n{{profile}}\n"""\
                """Based on the historical profiles provided, please generate a title for the given user's input abstract. """\
                    """Please generate it in the following format: {'title': 'generated title'} without explanation, and use only English.\n"""\
                        f"""{query} 'title':"""
        length_add = len(tokenizer("{{profile}}")['input_ids'])

        max_len_prompt = max_length + length_add - len(
            tokenizer(instruction)['input_ids'])
        per_p_max_length = max_len_prompt // len(profile)

        saved_tokens = 0
        prompts = []
        for p in profile:
            needed_part_len = len(
                tokenizer(f"'abstract': '' 'title': '{p['title']}'\n")
                ['input_ids'])
            tokens = tokenizer(p["abstract"],
                               max_length=per_p_max_length + saved_tokens -
                               needed_part_len,
                               truncation=True)
            saved_tokens += per_p_max_length - len(
                tokens['input_ids']) - needed_part_len
            new_asbtract = tokenizer.batch_decode([tokens['input_ids']],
                                                  skip_special_tokens=True)[0]
            prompt = f"'abstract': '{new_asbtract}' 'title': '{p['title']}'\n"
            prompts.append(prompt)

        instruction = instruction.replace("{{profile}}", "".join(prompts))
        return instruction


# LaMP_7
def get_prompt_LaMP_7(inp, profile, max_length, tokenizer):
    query = get_query_LaMP_7(inp)

    if len(profile) == 0:
        return """Please paraphrase the user's input tweet without any explanation before or after it.\n"""\
        """Please generate it in the following format: {'tweet': 'generated tweet'} without explanation, and use only English.\n"""\
            f"""{query}"""
    else:
        instruction = """The historical tweets are as follows:\n{{profile}}\n"""\
            """Based on the style pattern of the historical tweets provided, please paraphrase the user's input tweet without any explanation before or after it.\n"""\
            """Please generate it in the following format: {'tweet': 'generated tweet'} without explanation, and use only English.\n"""\
            f"""{query}"""
        length_add = len(tokenizer("{{profile}}")['input_ids'])

        max_len_prompt = max_length + length_add - len(
            tokenizer(instruction)['input_ids'])
        per_p_max_length = max_len_prompt // len(profile)

        saved_tokens = 0
        prompts = []
        for p in profile:
            needed_part_len = len(tokenizer(f"'tweet': ''\n")['input_ids'])
            tokens = tokenizer(p["text"],
                               max_length=per_p_max_length + saved_tokens -
                               needed_part_len,
                               truncation=True)
            saved_tokens += per_p_max_length - len(
                tokens['input_ids']) - needed_part_len
            new_asbtract = tokenizer.batch_decode([tokens['input_ids']],
                                                  skip_special_tokens=True)[0]
            prompt = f"'tweet': '{new_asbtract}'\n"
            prompts.append(prompt)

        instruction = instruction.replace("{{profile}}", "".join(prompts))
        return instruction


def load_get_prompt_fn(task):
    if task.startswith('LaMP_1'):
        return get_prompt_LaMP_1
    elif task.startswith('LaMP_2'):
        return get_prompt_LaMP_2
    elif task.startswith('LaMP_3'):
        return get_prompt_LaMP_3
    elif task.startswith('LaMP_4'):
        return get_prompt_LaMP_4
    elif task.startswith('LaMP_5'):
        return get_prompt_LaMP_5
    elif task.startswith('LaMP_7'):
        return get_prompt_LaMP_7
    else:
        raise ValueError('task error')
