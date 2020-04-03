# # This file is a spider script to get the images from google or some site.
# # I guess the manner of searching is key word maybe the image for the similar.
# # Begin: 23: 24 2020 / 1 / 9
# # I guess I will succeed in two days. Maybe
# # The next is the script.

# # import some necessary packages
# # Base Class
from abc import ABC, abstractmethod


class Spider(ABC):
    """
    Base class
    """

    def __init__(self, first_word="", *args):
        self.last_input = first_word  # # record of last search.
        self.response = ""

    @abstractmethod
    def get_response(self, *args):
        self.response = ""

    @abstractmethod
    def parse_response(self, *args):
        pass

    @abstractmethod
    def write_to_local(self, *args):
        pass

    @abstractmethod
    def run_spider(self):
        pass


# # run it
if __name__ == "__main__":
    print("Base class Spider is defined here")
