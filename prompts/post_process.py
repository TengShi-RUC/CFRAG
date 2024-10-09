def post_process_LaMP_1(results):
    return results


def post_process_LaMP_2(results):
    processed_results = []
    for generate_data in results:
        processed_predict = generate_data.strip('\'\"')
        processed_results.append(processed_predict)
    return processed_results


def post_process_LaMP_3(results):
    processed_results = []
    for generate_data in results:
        try:
            score = int(round(float(generate_data), 0))
            processed_predict = str(score)
        except:
            begin_index = generate_data.find("'score': ")
            if begin_index == -1:
                begin_index = generate_data.find("\"score\": ")
                if begin_index != -1:
                    begin_index += len("\"score\": ")
            else:
                begin_index += len("'score': ")
            if begin_index == -1:
                processed_predict = '3'
            else:
                end_index = generate_data[begin_index:].find('}') + begin_index
                processed_predict = generate_data[begin_index:end_index]

        processed_results.append(processed_predict)
    return processed_results


def post_process_LaMP_4(results):
    processed_results = []
    for generate_data in results:
        begin_index = generate_data.find("'title': '")
        if begin_index == -1:
            begin_index = generate_data.find("\"title\": \"")
            begin_index += len("\"title\": \"")
            end_index = generate_data[begin_index:].find("\"}") + begin_index
        else:
            begin_index += len("'title': '")
            end_index = generate_data[begin_index:].find("'}") + begin_index
        processed_predict = generate_data[begin_index:end_index]
        processed_results.append(processed_predict)
    return processed_results


def post_process_LaMP_5(results):
    processed_results = []
    for generate_data in results:
        begin_index = generate_data.find("'title': '")
        if begin_index == -1:
            begin_index = generate_data.find("\"title\": \"")
            begin_index += len("\"title\": \"")
            end_index = generate_data[begin_index:].find("\"}") + begin_index
        else:
            begin_index += len("'title': '")
            end_index = generate_data[begin_index:].find("'}") + begin_index
        processed_predict = generate_data[begin_index:end_index]
        processed_results.append(processed_predict)
    return processed_results


def post_process_LaMP_7(results):
    processed_results = []
    for generate_data in results:
        begin_index = generate_data.find("'tweet': '")
        if begin_index == -1:
            begin_index = generate_data.find("\"tweet\": \"")
            begin_index += len("\"tweet\": \"")
            end_index = generate_data[begin_index:].find("\"}") + begin_index
        else:
            begin_index += len("'tweet': '")
            end_index = generate_data[begin_index:].find("'}") + begin_index

        processed_predict = generate_data[begin_index:end_index]
        processed_results.append(processed_predict)
    return processed_results


def load_post_process_function(task):
    if task.startswith('LaMP_1'):
        return post_process_LaMP_1
    elif task.startswith('LaMP_2'):
        return post_process_LaMP_2
    elif task.startswith('LaMP_3'):
        return post_process_LaMP_3
    elif task.startswith('LaMP_4'):
        return post_process_LaMP_4
    elif task.startswith('LaMP_5'):
        return post_process_LaMP_5
    elif task.startswith('LaMP_7'):
        return post_process_LaMP_7
    else:
        raise ValueError('task error')
