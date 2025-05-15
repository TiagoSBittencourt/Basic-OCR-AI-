    # This is a test code (don't touch)
#for i in range(5, 50, 5):
#    nn = OCRNeuralNetwork(i, data_matrix, data_labels, train_indices, False)
#    performance = str(test(data_matrix, data_labels, test_indices, nn))
#    print("{i} Hidden Nodes: {val}".format(i=i, val=performance))
#
#def test(data_matrix, data_labels, test_indices, nn):
#    avg_sum = 0
#    for _ in range(100):
#        correct_guess_count = 0
#        for idx in test_indices:
#            sample = data_matrix[idx]
#            prediction = nn.predict(sample)
#            if data_labels[idx] == prediction:
#                correct_guess_count += 1
#        avg_sum += (correct_guess_count / float(len(test_indices)))
#    return avg_sum / 100