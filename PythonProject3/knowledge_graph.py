import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.node_types = defaultdict(list)

    def build_graph_from_data(self, data):
        """从数据构建知识图谱"""
        print("开始构建知识图谱...")

        # 添加电影节点
        for _, row in data.iterrows():
            movie_id = f"movie_{row['id']}"
            movie_title = row.get('title_x', 'Unknown')

            # 添加电影节点
            self.graph.add_node(movie_id,
                                type='movie',
                                title=movie_title,
                                year=self._extract_year(row.get('release_date', '')),
                                rating=row.get('vote_average', 0),
                                popularity=row.get('popularity', 0),
                                vote_count=row.get('vote_count', 0))

            # 添加类型节点和关系
            if 'genres_parsed' in row and row['genres_parsed']:
                for genre in row['genres_parsed']:
                    if genre:
                        genre_id = f"genre_{genre.replace(' ', '_').replace('/', '_')}"
                        self.graph.add_node(genre_id, type='genre', name=genre)
                        self.graph.add_edge(movie_id, genre_id, relation='has_genre')

            # 添加导演节点和关系
            if 'directors' in row and row['directors']:
                for director in row['directors']:
                    if director:
                        director_id = f"director_{director.replace(' ', '_').replace('/', '_')}"
                        self.graph.add_node(director_id, type='director', name=director)
                        self.graph.add_edge(movie_id, director_id, relation='directed_by')

            # 添加演员节点和关系
            if 'cast_parsed' in row and row['cast_parsed']:
                for actor in row['cast_parsed'][:5]:  # 只取前5个主要演员
                    if actor:
                        actor_id = f"actor_{actor.replace(' ', '_').replace('/', '_')}"
                        self.graph.add_node(actor_id, type='actor', name=actor)
                        self.graph.add_edge(movie_id, actor_id, relation='starring')

            # 添加关键词节点和关系
            if 'keywords_parsed' in row and row['keywords_parsed']:
                for keyword in row['keywords_parsed'][:5]:  # 只取前5个关键词
                    if keyword:
                        keyword_id = f"keyword_{keyword.replace(' ', '_').replace('/', '_')}"
                        self.graph.add_node(keyword_id, type='keyword', name=keyword)
                        self.graph.add_edge(movie_id, keyword_id, relation='has_keyword')

        print(f"知识图谱构建完成，包含 {self.graph.number_of_nodes()} 个节点和 {self.graph.number_of_edges()} 条边")

        # 统计节点类型
        for node, attrs in self.graph.nodes(data=True):
            self.node_types[attrs.get('type', 'unknown')].append(node)

        for node_type, nodes in self.node_types.items():
            print(f"{node_type} 节点: {len(nodes)} 个")

    def _extract_year(self, date_str):
        """从日期字符串中提取年份"""
        if pd.isna(date_str) or date_str == '':
            return 'Unknown'
        try:
            return str(date_str).split('-')[0]
        except:
            return 'Unknown'

    def get_movie_recommendations(self, user_input, top_n=10):
        """根据用户输入获取电影推荐"""
        recommendations = []

        # 查找相关节点
        related_nodes = []
        for node, attrs in self.graph.nodes(data=True):
            if any(str(user_input).lower() in str(value).lower()
                   for key, value in attrs.items() if key != 'type' and value is not None):
                related_nodes.append(node)

        if not related_nodes:
            return recommendations

        # 基于相关节点找到电影
        for node in related_nodes:
            # 如果是电影节点，直接推荐
            if self.graph.nodes[node].get('type') == 'movie':
                recommendations.append(node)
            else:
                # 如果是其他类型节点，找到相关的电影
                neighbors = list(self.graph.neighbors(node))
                for neighbor in neighbors:
                    if self.graph.nodes[neighbor].get('type') == 'movie':
                        recommendations.append(neighbor)

        # 去重并排序
        recommendations = list(set(recommendations))

        # 按评分和流行度综合排序
        scored_recommendations = []
        for movie_id in recommendations:
            movie_data = self.graph.nodes[movie_id]
            rating = movie_data.get('rating', 0)
            popularity = movie_data.get('popularity', 0)
            vote_count = movie_data.get('vote_count', 0)

            # 综合评分（评分权重0.6，流行度权重0.2，投票数权重0.2）
            normalized_popularity = min(popularity / 100, 10) if popularity > 0 else 0
            normalized_votes = min(vote_count / 1000, 10) if vote_count > 0 else 0
            score = rating * 0.6 + normalized_popularity * 0.2 + normalized_votes * 0.2
            scored_recommendations.append((movie_id, score))

        scored_recommendations.sort(key=lambda x: x[1], reverse=True)

        return [movie_id for movie_id, _ in scored_recommendations[:top_n]]

    def get_movie_details(self, movie_id):
        """获取电影的详细信息"""
        if movie_id in self.graph.nodes:
            return self.graph.nodes[movie_id]
        return None

    def get_graph_info(self):
        """获取知识图谱信息"""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'node_types': dict(self.node_types)
        }

    def find_movie_by_title(self, title):
        """根据标题查找电影"""
        for node, attrs in self.graph.nodes(data=True):
            if (attrs.get('type') == 'movie' and
                    str(title).lower() in str(attrs.get('title', '')).lower()):
                return node
        return None

    def visualize_graph(self, filename='knowledge_graph.png'):
        """可视化知识图谱（简化版，仅显示部分节点）"""
        try:
            plt.figure(figsize=(20, 15))

            # 只显示部分节点以避免过于拥挤
            nodes_to_show = []
            for node, attrs in self.graph.nodes(data=True):
                if attrs.get('type') == 'movie':
                    nodes_to_show.append(node)
                    if len(nodes_to_show) >= 15:  # 限制显示的电影数量
                        break

            # 添加与这些电影相关的其他节点
            subgraph_nodes = set(nodes_to_show)
            for node in nodes_to_show:
                neighbors = list(self.graph.neighbors(node))
                subgraph_nodes.update(neighbors[:5])  # 每个电影只显示前5个相关节点

            subgraph = self.graph.subgraph(subgraph_nodes)

            # 设置节点颜色和大小
            node_colors = []
            node_sizes = []
            labels = {}

            for node in subgraph.nodes():
                node_type = self.graph.nodes[node].get('type')
                if node_type == 'movie':
                    node_colors.append('lightblue')
                    node_sizes.append(800)
                    labels[node] = self.graph.nodes[node].get('title', '')[:15]
                elif node_type == 'genre':
                    node_colors.append('lightgreen')
                    node_sizes.append(500)
                    labels[node] = self.graph.nodes[node].get('name', '')[:10]
                elif node_type == 'director':
                    node_colors.append('lightcoral')
                    node_sizes.append(600)
                    labels[node] = self.graph.nodes[node].get('name', '')[:10]
                elif node_type == 'actor':
                    node_colors.append('gold')
                    node_sizes.append(400)
                    labels[node] = self.graph.nodes[node].get('name', '')[:10]
                elif node_type == 'keyword':
                    node_colors.append('lightyellow')
                    node_sizes.append(300)
                    labels[node] = self.graph.nodes[node].get('name', '')[:8]
                else:
                    node_colors.append('gray')
                    node_sizes.append(300)
                    labels[node] = node[:8]

            # 绘制图形
            pos = nx.spring_layout(subgraph, k=2, iterations=100)
            nx.draw(subgraph, pos,
                    node_color=node_colors,
                    node_size=node_sizes,
                    labels=labels,
                    font_size=8,
                    font_weight='bold',
                    edge_color='gray',
                    alpha=0.8,
                    linewidths=0.5)

            plt.title("TMDB电影知识图谱 (部分显示)", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"知识图谱已保存为 {filename}")
        except Exception as e:
            print(f"可视化知识图谱时出错: {e}")
            import traceback
            traceback.print_exc()