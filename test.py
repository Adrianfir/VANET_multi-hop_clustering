import bisect

class SortedList:
    def __init__(self):
        self.list = []

    def insert(self, num):
        # Insert the negative of the number to maintain a sorted list in descending order
        bisect.insort(self.list, -num)

    def get_sorted_list(self):
        # Return the list with numbers negated back to original
        return [-num for num in self.list]

# Usage:
sorted_list = SortedList()
sorted_list.insert(10)
sorted_list.insert(5)
sorted_list.insert(20)

print(sorted_list.get_sorted_list())  # Output: [20, 10, 5]
