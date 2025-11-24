import pandas as pd
import os
import ast


class DataLoader:
    def __init__(self, movies_file_path, credits_file_path):
        self.movies_file_path = movies_file_path
        self.credits_file_path = credits_file_path
        self.data = None

    def load_data(self):
        """从两个CSV文件加载并合并数据"""
        try:
            # 检查文件是否存在
            if not os.path.exists(self.movies_file_path):
                raise FileNotFoundError(f"电影数据文件 {self.movies_file_path} 不存在")
            if not os.path.exists(self.credits_file_path):
                raise FileNotFoundError(f"演职员数据文件 {self.credits_file_path} 不存在")

            print("正在加载电影数据...")
            movies_df = pd.read_csv(self.movies_file_path)
            print(f"成功加载电影数据，共 {len(movies_df)} 条记录")

            print("正在加载演职员数据...")
            credits_df = pd.read_csv(self.credits_file_path)
            print(f"成功加载演职员数据，共 {len(credits_df)} 条记录")

            # 合并数据
            print("正在合并数据...")
            self.data = self._merge_data(movies_df, credits_df)
            print(f"数据合并完成，共 {len(self.data)} 条记录")

            return self.data

        except Exception as e:
            print(f"加载数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _merge_data(self, movies_df, credits_df):
        """合并电影数据和演职员数据"""
        # 根据电影ID合并
        merged_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='inner')

        # 处理JSON格式的数据
        merged_df = self._process_json_columns(merged_df)

        return merged_df

    def _process_json_columns(self, df):
        """处理JSON格式的列"""
        print("正在处理JSON格式的列...")

        # 解析类型
        if 'genres' in df.columns:
            df['genres_parsed'] = df['genres'].apply(self._parse_genres)
            print("成功解析类型数据")

        # 解析关键词
        if 'keywords' in df.columns:
            df['keywords_parsed'] = df['keywords'].apply(self._parse_keywords)
            print("成功解析关键词数据")

        # 解析演员
        if 'cast' in df.columns:
            df['cast_parsed'] = df['cast'].apply(self._parse_cast)
            print("成功解析演员数据")

        # 解析工作人员
        if 'crew' in df.columns:
            df['crew_parsed'] = df['crew'].apply(self._parse_crew)
            df['directors'] = df['crew_parsed'].apply(self._extract_directors)
            print("成功解析工作人员数据")

        # 解析制作公司
        if 'production_companies' in df.columns:
            df['production_companies_parsed'] = df['production_companies'].apply(self._parse_production_companies)
            print("成功解析制作公司数据")

        return df

    def _parse_genres(self, json_str):
        """解析类型信息"""
        if pd.isna(json_str) or json_str == '' or json_str == '[]':
            return []
        try:
            data = ast.literal_eval(json_str)
            if isinstance(data, list):
                genres = []
                for item in data:
                    if isinstance(item, dict) and 'name' in item:
                        genres.append(item['name'])
                return genres
            return []
        except Exception as e:
            print(f"解析类型时出错: {e}")
            return []

    def _parse_keywords(self, json_str):
        """解析关键词信息"""
        if pd.isna(json_str) or json_str == '' or json_str == '[]':
            return []
        try:
            data = ast.literal_eval(json_str)
            if isinstance(data, list):
                keywords = []
                for item in data:
                    if isinstance(item, dict) and 'name' in item:
                        keywords.append(item['name'])
                return keywords[:10]  # 只取前10个关键词
            return []
        except Exception as e:
            print(f"解析关键词时出错: {e}")
            return []

    def _parse_cast(self, json_str):
        """解析演员信息"""
        if pd.isna(json_str) or json_str == '' or json_str == '[]':
            return []
        try:
            data = ast.literal_eval(json_str)
            if isinstance(data, list):
                actors = []
                for item in data[:8]:  # 只取前8个主要演员
                    if isinstance(item, dict) and 'name' in item:
                        actors.append(item['name'])
                return actors
            return []
        except Exception as e:
            print(f"解析演员信息时出错: {e}")
            return []

    def _parse_crew(self, json_str):
        """解析工作人员信息"""
        if pd.isna(json_str) or json_str == '' or json_str == '[]':
            return []
        try:
            data = ast.literal_eval(json_str)
            if isinstance(data, list):
                return data
            return []
        except Exception as e:
            print(f"解析工作人员信息时出错: {e}")
            return []

    def _parse_production_companies(self, json_str):
        """解析制作公司信息"""
        if pd.isna(json_str) or json_str == '' or json_str == '[]':
            return []
        try:
            data = ast.literal_eval(json_str)
            if isinstance(data, list):
                companies = []
                for item in data:
                    if isinstance(item, dict) and 'name' in item:
                        companies.append(item['name'])
                return companies
            return []
        except Exception as e:
            print(f"解析制作公司时出错: {e}")
            return []

    def _extract_directors(self, crew_list):
        """从工作人员列表中提取导演"""
        if not crew_list:
            return []
        directors = []
        for person in crew_list:
            if isinstance(person, dict) and person.get('job') == 'Director':
                name = person.get('name', '')
                if name:
                    directors.append(name)
        return directors

    def get_sample_data(self, n=5):
        """获取数据样本"""
        if self.data is not None:
            return self.data.head(n)
        return None

    def get_columns(self):
        """获取数据列名"""
        if self.data is not None:
            return self.data.columns.tolist()
        return []

    def get_movie_titles(self):
        """获取所有电影标题"""
        if self.data is not None and 'title_x' in self.data.columns:
            titles = self.data['title_x'].tolist()
            # 过滤掉空值
            titles = [title for title in titles if pd.notna(title) and str(title).strip()]
            return titles
        return []

    def display_data_info(self):
        """显示数据信息"""
        if self.data is not None:
            print("\n数据信息:")
            print(f"总记录数: {len(self.data)}")
            print(f"列数: {len(self.data.columns)}")

            # 显示前5条记录
            display_cols = ['title_x', 'release_date', 'vote_average', 'popularity']
            available_cols = [col for col in display_cols if col in self.data.columns]

            print(f"\n前5条记录:")
            print(self.data[available_cols].head())

            # 显示一些统计信息
            if 'genres_parsed' in self.data.columns:
                all_genres = [genre for sublist in self.data['genres_parsed'] for genre in sublist]
                print(f"\n总类型数: {len(set(all_genres))}")
                print(f"最常见的5个类型: {pd.Series(all_genres).value_counts().head(5).to_dict()}")

            if 'directors' in self.data.columns:
                all_directors = [director for sublist in self.data['directors'] for director in sublist]
                print(f"总导演数: {len(set(all_directors))}")
                print(f"作品最多的5位导演: {pd.Series(all_directors).value_counts().head(5).to_dict()}")

            if 'cast_parsed' in self.data.columns:
                all_actors = [actor for sublist in self.data['cast_parsed'] for actor in sublist]
                print(f"总演员数: {len(set(all_actors))}")
                print(f"出演作品最多的5位演员: {pd.Series(all_actors).value_counts().head(5).to_dict()}")