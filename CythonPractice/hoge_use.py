from hoge import func1  # とか
import hoge  # とかすることで、　(ただし、func2は使えません。)

if __name__ == "__main__":
    print(hoge.func1(1e7))
