class MovieRecommender:
    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def recommend_by_keyword(self, keyword, top_n=10):
        """基于关键词推荐电影"""
        return self.kg.get_movie_recommendations(keyword, top_n)

    def recommend_by_movie(self, movie_title, top_n=10):
        """基于电影推荐相似电影"""
        # 首先找到该电影
        movie_node = self.kg.find_movie_by_title(movie_title)

        if not movie_node:
            return []

        # 获取该电影的特征
        movie_features = set()
        neighbors = list(self.kg.graph.neighbors(movie_node))
        for neighbor in neighbors:
            neighbor_attrs = self.kg.graph.nodes[neighbor]
            if neighbor_attrs.get('type') in ['genre', 'director', 'actor', 'keyword']:
                movie_features.add(neighbor)

        # 寻找具有相似特征的电影
        similar_movies = []
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'movie' and node != movie_node:
                # 计算相似度
                node_features = set(self.kg.graph.neighbors(node))
                common_features = len(movie_features.intersection(node_features))
                if common_features > 0:
                    similarity = common_features / len(movie_features.union(node_features))
                    # 考虑评分因素
                    rating_similarity = min(attrs.get('rating', 0) / 10, 1.0)
                    combined_similarity = similarity * 0.7 + rating_similarity * 0.3
                    similar_movies.append((node, combined_similarity))

        # 按相似度排序
        similar_movies.sort(key=lambda x: x[1], reverse=True)

        return [movie_id for movie_id, _ in similar_movies[:top_n]]

    def get_recommendation_details(self, movie_ids):
        """获取推荐电影的详细信息"""
        details = []
        for movie_id in movie_ids:
            movie_info = self.kg.get_movie_details(movie_id)
            if movie_info:
                # 获取相关类型、导演和演员
                genres = []
                directors = []
                actors = []
                keywords = []

                neighbors = list(self.kg.graph.neighbors(movie_id))
                for neighbor in neighbors:
                    neighbor_attrs = self.kg.graph.nodes[neighbor]
                    if neighbor_attrs.get('type') == 'genre':
                        genres.append(neighbor_attrs.get('name', 'Unknown'))
                    elif neighbor_attrs.get('type') == 'director':
                        directors.append(neighbor_attrs.get('name', 'Unknown'))
                    elif neighbor_attrs.get('type') == 'actor':
                        actors.append(neighbor_attrs.get('name', 'Unknown'))
                    elif neighbor_attrs.get('type') == 'keyword':
                        keywords.append(neighbor_attrs.get('name', 'Unknown'))

                movie_info['genres_list'] = genres
                movie_info['directors_list'] = directors
                movie_info['actors_list'] = actors[:5]  # 只显示前5个演员
                movie_info['keywords_list'] = keywords[:5]  # 只显示前5个关键词

                details.append(movie_info)

        return details

    def display_recommendations(self, recommendations):
        """格式化显示推荐结果"""
        if not recommendations:
            print("未找到相关推荐")
            return

        print(f"\n为您找到 {len(recommendations)} 部推荐电影：")
        print("=" * 120)

        for i, movie in enumerate(recommendations, 1):
            rating = movie.get('rating', 'N/A')
            popularity = movie.get('popularity', 'N/A')
            vote_count = movie.get('vote_count', 'N/A')

            # 创建评分星号
            if isinstance(rating, (int, float)):
                stars_count = int(rating)
                rating_stars = '★' * stars_count + '☆' * (10 - stars_count)
                rating_display = f"{rating:.1f}/10 {rating_stars}"
            else:
                rating_display = 'N/A'

            print(f"{i}. {movie.get('title', '未知标题')} ({movie.get('year', '未知年份')})")
            print(f"   评分: {rating_display}")
            print(f"   流行度: {popularity} | 投票数: {vote_count}")
            print(f"   类型: {', '.join(movie.get('genres_list', ['未知']))}")
            print(f"   导演: {', '.join(movie.get('directors_list', ['未知']))}")
            print(f"   主演: {', '.join(movie.get('actors_list', ['未知']))}")
            if movie.get('keywords_list'):
                print(f"   关键词: {', '.join(movie.get('keywords_list', []))}")
            print("-" * 120)