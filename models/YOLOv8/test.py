from ultralytics import YOLO
import os
import numpy as np


model = YOLO("best.pt")
cls = model.names

# Get the current file's directory
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

# Go up one level to the parent directory
parent_directory = os.path.dirname(current_directory)
test_dir = os.path.join(parent_directory, 'dataset', 'test')

# row - true
# col - prediction
confusion_matrix = np.zeros(shape=(len(cls), len(cls)))

for ind, c in cls.items():
	cDir = os.path.join(test_dir, c)
	rez = model.predict(cDir)
	for r in rez:
		p = r.probs.top1
		confusion_matrix[ind, p] += 1

print("%-20s|   p   |   r   |   f1   "%"Class")

matSum = np.sum(confusion_matrix)
f1Scores = 0
pScores = 0
rScores = 0

tTP = 0
tFP = 0
tFN = 0

for ind, c in cls.items():
	cTP = confusion_matrix[ind, ind]
	cFP = np.sum(confusion_matrix[:, ind]) - cTP
	cFN = np.sum(confusion_matrix[ind]) - cTP
	cTN = matSum - cTP - cFP - cFN

	tTP += cTP
	tFP += cFP
	tFN += cFN

	precision = cTP / (cTP + cFP)
	recall = cTP / (cTP + cFN)

	pScores += precision
	rScores += recall

	f1 = 2 * precision * recall / (precision + recall)
	f1Scores += f1

	print("%-20s| %5.4f | %5.4f | %6.4f"%(c, precision, recall, f1))

print()
print("Makro")
print("p=", pScores / len(cls))
print("r=", rScores / len(cls))
print("f1=", f1Scores / len(cls))
print()
print("Mikro")
precision = tTP / (tTP + tFP)
recall = tTP / (tTP + tFN)
f1 = 2 * precision * recall / (precision + recall)

print("p=", precision)
print("r=", recall)
print("f1=", f1)
