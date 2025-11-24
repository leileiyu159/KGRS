import pandas as pd
import os


def check_columns():
    """检查数据文件的列名"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    movies_file = os.path.join(current_dir, 'tmdb_5000_movies.csv')
    credits_file = os.path.join(current_dir, 'tmdb_5000_credits.csv')

    print("检查电影数据文件列名:")
    if os.path.exists(movies_file):
        movies_df = pd.read_csv(movies_file)
        print(f"电影数据列名: {list(movies_df.columns)}")
        print(f"电影数据形状: {movies_df.shape}")
        print("\n前3行数据:")
        print(movies_df.head(3))
    else:
        print(f"电影数据文件不存在: {movies_file}")

    print("\n" + "=" * 50 + "\n")

    print("检查演职员数据文件列名:")
    if os.path.exists(credits_file):
        credits_df = pd.read_csv(credits_file)
        print(f"演职员数据列名: {list(credits_df.columns)}")
        print(f"演职员数据形状: {credits_df.shape}")
        print("\n前3行数据:")
        print(credits_df.head(3))
    else:
        print(f"演职员数据文件不存在: {credits_file}")


if __name__ == "__main__":
    check_columns()