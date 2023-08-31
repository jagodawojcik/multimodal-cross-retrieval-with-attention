import numpy as np

RANK_K = 5

def prepare_data_for_evaluation(embeddings):
    labels = []
    embeddings_list = []
    for label, embedding in embeddings.items():
        labels.extend([label]*len(embedding))
        embeddings_list.extend(embedding)
    return np.array(embeddings_list), np.array(labels)

def average_precision(relevant_scores, k):
    if len(relevant_scores) == 0:
        return 0.0

    score = 0.0
    num_hits = 0.0

    for i, relevant in enumerate(relevant_scores[:k]):
        if relevant > 0:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(relevant_scores), k)

def calculate_MAP(queries, retrievals, query_labels, retrieval_labels):
    APs = []
    for i in range(len(queries)):
        query = queries[i]
        current_query_label = query_labels[i]

        # calculate the distance between the query and all retrievals
        distances = np.linalg.norm(retrievals - query, axis=1)

        # sort the distances to find the closest and get their indices
        closest_indices = np.argsort(distances)

        # get the labels for the closest instances
        closest_labels = retrieval_labels[closest_indices]

        # determine which retrievals are true positives
        relevant_retrievals = (closest_labels == current_query_label)

        # calculate the average precision for this query
        ap = average_precision(relevant_retrievals, RANK_K)
        APs.append(ap)

    # the mean average precision is the mean of the average precision values for all queries
    return np.mean(APs)

def evaluate(visual_test_embeddings, tactile_test_embeddings, visual_train_embeddings, tactile_train_embeddings):
    visual_test_queries, visual_test_query_labels = prepare_data_for_evaluation(visual_test_embeddings)
    tactile_test_queries, tactile_test_query_labels = prepare_data_for_evaluation(tactile_test_embeddings)
    visual_train_queries, visual_train_query_labels = prepare_data_for_evaluation(visual_train_embeddings)
    tactile_train_queries, tactile_train_query_labels = prepare_data_for_evaluation(tactile_train_embeddings)

    # Calculate MAP for tactile2visual retrieval
    MAP_tactile2visual = calculate_MAP(tactile_test_queries, visual_train_queries, tactile_test_query_labels, visual_train_query_labels)

    # Calculate MAP for visual2tactile retrieval
    MAP_visual2tactile = calculate_MAP(visual_test_queries, tactile_train_queries, visual_test_query_labels, tactile_train_query_labels)

    return MAP_tactile2visual, MAP_visual2tactile

if __name__ == "__main__":
    visual_embeddings = np.load("visual_embeddings_kaggle_train.npy", allow_pickle=True).item()
    tactile_embeddings = np.load("tactile_embeddings_kaggle_train.npy", allow_pickle=True).item()
    visual_embeddings_test = np.load("visual_embeddings_kaggle_test.npy", allow_pickle=True).item()  
    tactile_embeddings_test = np.load("tactile_embeddings_kaggle_test.npy", allow_pickle=True).item() 
    MAP_tactile2visual, MAP_visual2tactile = evaluate_audio_vis(visual_embeddings_test, tactile_embeddings_test, visual_embeddings, tactile_embeddings)
    print(MAP_tactile2visual, MAP_visual2tactile)