exam_scores = [78, 88, 90, 64, 95, 61, 50, 66, 40, 77, 71]

A = 80

B = 75

C = 70

D = 60

A_list, B_list, C_list, D_list, F_list = [], [], [], [], []


for mark in exam_scores:
    if mark >= A:
        A_list.append(mark)
    elif mark >= B:
        B_list.append(mark)
    elif mark >= C:
        C_list.append(mark)
    elif mark >= D:
        D_list.append(mark)
    else:
        F_list.append(mark)


print(f"These exam scores achieved a mark of A: {A_list}")
print(f"These exam scores achieved a mark of B: {B_list}")
print(f"These exam scores achieved a mark of C: {C_list}")
print(f"These exam scores achieved a mark of D: {D_list}")
print(f"These exam scores all failed: {F_list}")


total_score = sum(exam_scores)
average_score = total_score/len(exam_scores)
print(f"The average score of all exam scores is: {average_score:.2f}")
