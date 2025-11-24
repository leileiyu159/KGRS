import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from knowledge_graph import KnowledgeGraph
from recommender import MovieRecommender


class MovieRecommendationSystem:
    def __init__(self):
        self.data_loader = None
        self.knowledge_graph = None
        self.recommender = None
        self.initialized = False

    def initialize(self, movies_file_path, credits_file_path):
        """初始化系统"""
        try:
            print("正在初始化电影推荐系统...")

            # 加载数据
            self.data_loader = DataLoader(movies_file_path, credits_file_path)
            data = self.data_loader.load_data()

            if data is None:
                print("数据加载失败！")
                return False

            # 显示数据信息
            self.data_loader.display_data_info()

            # 构建知识图谱
            self.knowledge_graph = KnowledgeGraph()
            self.knowledge_graph.build_graph_from_data(data)

            # 创建推荐器
            self.recommender = MovieRecommender(self.knowledge_graph)

            # 生成知识图谱可视化
            self.knowledge_graph.visualize_graph('knowledge_graph.png')

            self.initialized = True
            print("系统初始化成功！")
            return True

        except Exception as e:
            print(f"系统初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def display_system_info(self):
        """显示系统信息"""
        if not self.initialized:
            print("系统未初始化")
            return

        graph_info = self.knowledge_graph.get_graph_info()
        print("\n" + "=" * 50)
        print("TMDB电影推荐系统")
        print("=" * 50)
        print(f"知识图谱统计:")
        print(f"  - 总节点数: {graph_info['nodes']}")
        print(f"  - 总边数: {graph_info['edges']}")

        for node_type, nodes in graph_info['node_types'].items():
            print(f"  - {node_type}节点: {len(nodes)}")
        print("=" * 50)

    def display_available_movies(self):
        """显示可用的电影列表"""
        if not self.initialized:
            print("系统未初始化")
            return

        movies = self.data_loader.get_movie_titles()
        if movies:
            print("\n可用电影列表 (前20部):")
            for i, movie in enumerate(movies[:20], 1):
                print(f"  {i}. {movie}")
            if len(movies) > 20:
                print(f"  ... 还有 {len(movies) - 20} 部电影")

            print(f"\n提示: 您可以使用电影名称、导演、演员、类型或关键词进行搜索")
        else:
            print("未找到电影数据")

    def run_interactive_mode(self):
        """运行交互模式"""
        if not self.initialized:
            print("系统未初始化，无法运行")
            return

        while True:
            print("\n" + "=" * 50)
            print("TMDB电影推荐系统")
            print("=" * 50)
            print("请选择推荐方式:")
            print("1. 基于关键词推荐 (导演、演员、类型、关键词等)")
            print("2. 基于电影推荐相似电影")
            print("3. 显示可用电影列表")
            print("4. 显示系统信息")
            print("5. 退出系统")
            print("=" * 50)

            choice = input("请输入选择 (1-5): ").strip()

            if choice == '1':
                self.keyword_recommendation()
            elif choice == '2':
                self.movie_recommendation()
            elif choice == '3':
                self.display_available_movies()
            elif choice == '4':
                self.display_system_info()
            elif choice == '5':
                print("感谢使用TMDB电影推荐系统，再见！")
                break
            else:
                print("无效选择，请重新输入")

    def keyword_recommendation(self):
        """基于关键词的推荐"""
        keyword = input("请输入关键词 (导演、演员、类型、关键词等): ").strip()
        if not keyword:
            print("关键词不能为空")
            return

        try:
            top_n_input = input("请输入推荐数量 (默认10): ").strip()
            top_n = int(top_n_input) if top_n_input else 10
        except ValueError:
            top_n = 10
            print("输入无效，使用默认值10")

        print(f"\n正在基于关键词 '{keyword}' 搜索推荐电影...")
        movie_ids = self.recommender.recommend_by_keyword(keyword, top_n)
        recommendations = self.recommender.get_recommendation_details(movie_ids)
        self.recommender.display_recommendations(recommendations)

    def movie_recommendation(self):
        """基于电影的推荐"""
        movie_title = input("请输入电影名称: ").strip()
        if not movie_title:
            print("电影名称不能为空")
            return

        # 检查电影是否存在
        movie_node = self.knowledge_graph.find_movie_by_title(movie_title)
        if not movie_node:
            print(f"未找到电影 '{movie_title}'，请检查名称是否正确")
            self.display_available_movies()
            return

        try:
            top_n_input = input("请输入推荐数量 (默认10): ").strip()
            top_n = int(top_n_input) if top_n_input else 10
        except ValueError:
            top_n = 10
            print("输入无效，使用默认值10")

        print(f"\n正在寻找与 '{movie_title}' 相似的电影...")
        movie_ids = self.recommender.recommend_by_movie(movie_title, top_n)
        recommendations = self.recommender.get_recommendation_details(movie_ids)
        self.recommender.display_recommendations(recommendations)


def main():
    """主函数"""
    # 数据文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    movies_file = os.path.join(current_dir, 'tmdb_5000_movies.csv')
    credits_file = os.path.join(current_dir, 'tmdb_5000_credits.csv')

    # 检查文件是否存在
    if not os.path.exists(movies_file):
        print(f"错误：电影数据文件不存在: {movies_file}")
        return

    if not os.path.exists(credits_file):
        print(f"错误：演职员数据文件不存在: {credits_file}")
        return

    # 创建并初始化系统
    system = MovieRecommendationSystem()

    if not system.initialize(movies_file, credits_file):
        print("系统初始化失败，请检查数据文件！")
        return

    # 显示系统信息
    system.display_system_info()

    # 运行交互模式
    system.run_interactive_mode()


if __name__ == "__main__":
    main()