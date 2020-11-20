import random


class ReplayBuffer:
    """
    Replay buffer object that stores elements up until a certain maximum size.

    N.B. Most of the code here can be reused in your TP2.
    """

    def __init__(self, buffer_size):
        """
        Init the buffer and store buffer_size property.
        """
        self.__buffer_size = buffer_size
        self.data = []

    def store(self, element):
        """
        Stores an element.

        If the buffer is already full, pop the oldest element inside.
        """
        self.data.append(element)

        if len(self.data) > self.__buffer_size:
            del self.data[0]

    def get_batch(self, batch_size):
        """
        Randomly samples batch_size elements from the buffer.

        Returns the list of sampled elements.
        """
        return random.sample(self.data, batch_size)