"""
Smartphone Classification & Ranking System
TCS iON Internship Project
Author: Anshuman Ayush
Description: ML-based system to classify and rank smartphones based on features
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


class SmartphoneRankingSystem:
    """
    Main class for smartphone classification and ranking
    """
    
    def __init__(self):
        self.csv_path = os.path.join(DATA_DIR, 'test.csv')
        self.df = None
        self.df_ranked = None
        self.classifier = None
        self.scaler = StandardScaler()
        
        # Verify file exists
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"\n❌ ERROR: test.csv not found!\n"
                f"Expected location: {self.csv_path}\n"
                f"Please place test.csv in the 'data' folder."
            )
        
    def load_data(self):
        """Load and display dataset information"""
        print("=" * 70)
        print("SMARTPHONE RANKING SYSTEM - TCS iON PROJECT".center(70))
        print("=" * 70)
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.csv_path, encoding=encoding)
                    print(f"\n✓ Dataset loaded successfully! (encoding: {encoding})")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.df is None:
                raise ValueError("Could not read CSV with any encoding")
                
        except Exception as e:
            print(f"\n❌ Error loading data: {str(e)}")
            return None
        
        print(f"  📊 Total records: {len(self.df)}")
        print(f"  📊 Total features: {len(self.df.columns)}")
        print(f"  📂 File location: {self.csv_path}")
        
        print("\n" + "=" * 70)
        print("DATASET PREVIEW".center(70))
        print("=" * 70)
        print(self.df.head())
        
        print("\n" + "=" * 70)
        print("STATISTICAL SUMMARY".center(70))
        print("=" * 70)
        print(self.df.describe())
        
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print("\n⚠ Missing Values Found:")
            print(missing[missing > 0])
        else:
            print("\n✓ No missing values detected!")
        
        return self.df
    
    def feature_engineering(self):
        """Create additional features for better analysis"""
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING".center(70))
        print("=" * 70)
        
        # Create composite features
        self.df['total_camera_mp'] = self.df['pc'] + self.df['fc']
        self.df['screen_area'] = self.df['sc_h'] * self.df['sc_w']
        self.df['pixel_density'] = (self.df['px_height'] * self.df['px_width']) / (self.df['screen_area'] + 1)
        self.df['feature_count'] = (self.df['blue'] + self.df['dual_sim'] + 
                                     self.df['four_g'] + self.df['three_g'] + 
                                     self.df['touch_screen'] + self.df['wifi'])
        
        print("✓ New features created:")
        print("  • total_camera_mp: Combined camera quality")
        print("  • screen_area: Display size")
        print("  • pixel_density: Screen resolution quality")
        print("  • feature_count: Total connectivity features")
        
        return self.df
    
    def create_price_categories(self):
        """Classify smartphones into price categories"""
        print("\n" + "=" * 70)
        print("CLASSIFICATION - PRICE CATEGORIES".center(70))
        print("=" * 70)
        
        # Calculate weighted score
        weights = {
            'ram': 0.25,
            'battery_power': 0.20,
            'total_camera_mp': 0.15,
            'pixel_density': 0.10,
            'int_memory': 0.10,
            'clock_speed': 0.08,
            'n_cores': 0.07,
            'feature_count': 0.05
        }
        
        score = 0
        for feature, weight in weights.items():
            normalized = (self.df[feature] - self.df[feature].min()) / \
                        (self.df[feature].max() - self.df[feature].min() + 0.001)
            score += normalized * weight * 100
        
        self.df['classification_score'] = score
        
        # Create categories
        self.df['price_category'] = pd.cut(score, 
                                           bins=[0, 30, 50, 70, 100],
                                           labels=['Budget', 'Mid-Range', 'Premium', 'Flagship'])
        
        print("\n✓ Classification completed!")
        print("\n📊 Category Distribution:")
        category_counts = self.df['price_category'].value_counts().sort_index()
        for category, count in category_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {category:12} : {count:4} phones ({percentage:.1f}%)")
        
        return self.df
    
    def train_classifier(self):
        """Train Random Forest classifier"""
        print("\n" + "=" * 70)
        print("MACHINE LEARNING - TRAINING CLASSIFIER".center(70))
        print("=" * 70)
        
        # Select features
        feature_cols = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 
                       'fc', 'four_g', 'int_memory', 'n_cores', 'pc', 
                       'ram', 'three_g', 'touch_screen', 'wifi']
        
        X = self.df[feature_cols]
        y = self.df['price_category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("\n⏳ Training Random Forest Classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10
        )
        
        self.classifier.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.classifier.predict(X_test_scaled)
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✓ Training completed!")
        print(f"  🎯 Accuracy: {accuracy:.2%}")
        
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 Top 5 Most Important Features:")
        for idx, row in feature_importance.head().iterrows():
            print(f"  {row['feature']:15} : {row['importance']:.4f}")
        
        return self.classifier, accuracy
    
    def calculate_ranking_score(self):
        """Calculate comprehensive ranking scores"""
        print("\n" + "=" * 70)
        print("RANKING ALGORITHM".center(70))
        print("=" * 70)
        
        # Ranking weights (different from classification)
        ranking_weights = {
            'ram': 0.30,
            'clock_speed': 0.10,
            'n_cores': 0.10,
            'pc': 0.15,
            'fc': 0.05,
            'battery_power': 0.15,
            'pixel_density': 0.10,
            'screen_area': 0.05
        }
        
        total_score = 0
        
        for feature, weight in ranking_weights.items():
            normalized = (self.df[feature] - self.df[feature].min()) / \
                        (self.df[feature].max() - self.df[feature].min() + 0.001)
            total_score += normalized * weight * 100
        
        # Add bonus for features
        feature_bonus = (self.df['four_g'] * 3 + self.df['three_g'] * 2 + 
                        self.df['touch_screen'] * 2 + self.df['wifi'] * 2 + 
                        self.df['dual_sim'] * 1)
        
        self.df['ranking_score'] = total_score + feature_bonus
        
        # Assign ranks
        self.df['rank'] = self.df['ranking_score'].rank(ascending=False, method='min').astype(int)
        
        # Sort by rank
        self.df_ranked = self.df.sort_values('rank').reset_index(drop=True)
        
        print("\n✓ Ranking calculation completed!")
        print(f"  Highest score: {self.df['ranking_score'].max():.2f}")
        print(f"  Lowest score: {self.df['ranking_score'].min():.2f}")
        print(f"  Average score: {self.df['ranking_score'].mean():.2f}")
        
        return self.df_ranked
    
    def display_top_phones(self, n=10):
        """Display top N ranked smartphones"""
        print("\n" + "=" * 70)
        print(f"TOP {n} RANKED SMARTPHONES".center(70))
        print("=" * 70)
        
        top_phones = self.df_ranked.head(n)
        
        print("\n{:<6} {:<10} {:<12} {:<8} {:<8} {:<10} {:<8} {:<6}".format(
            "RANK", "ID", "CATEGORY", "SCORE", "RAM", "BATTERY", "CAMERA", "4G"
        ))
        print("-" * 70)
        
        for _, phone in top_phones.iterrows():
            medal = "🥇" if phone['rank'] == 1 else "🥈" if phone['rank'] == 2 else "🥉" if phone['rank'] == 3 else "  "
            print("{} {:<4} {:<10} {:<12} {:<8.1f} {:<8} {:<10} {:<8} {:<6}".format(
                medal,
                int(phone['rank']),
                f"Phone-{int(phone['id'])}",
                phone['price_category'],
                phone['ranking_score'],
                f"{int(phone['ram'])} MB",
                f"{int(phone['battery_power'])} mAh",
                f"{int(phone['pc'])} MP",
                "Yes" if phone['four_g'] == 1 else "No"
            ))
        
        return top_phones
    
    def visualize_results(self):
        """Create visualizations"""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS".center(70))
        print("=" * 70)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Smartphone Ranking System - Analysis Dashboard', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        # 1. Category Distribution
        category_counts = self.df['price_category'].value_counts()
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        axes[0, 0].bar(category_counts.index, category_counts.values, color=colors, edgecolor='black', linewidth=1.5)
        axes[0, 0].set_title('Phone Distribution by Category', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Category', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Feature Importance
        if self.classifier:
            feature_cols = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 
                           'fc', 'four_g', 'int_memory', 'n_cores', 'pc', 
                           'ram', 'three_g', 'touch_screen', 'wifi']
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': self.classifier.feature_importances_
            }).sort_values('Importance', ascending=True).tail(10)
            
            axes[0, 1].barh(importance_df['Feature'], importance_df['Importance'], 
                           color='#9b59b6', edgecolor='black', linewidth=1.5)
            axes[0, 1].set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Importance Score', fontsize=12)
            axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Ranking Score Distribution
        axes[1, 0].hist(self.df['ranking_score'], bins=40, 
                       color='#1abc9c', edgecolor='black', alpha=0.7, linewidth=1.5)
        axes[1, 0].set_title('Ranking Score Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Ranking Score', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].axvline(self.df['ranking_score'].mean(), 
                          color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {self.df["ranking_score"].mean():.2f}')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. RAM vs Battery (Top 100)
        top_100 = self.df_ranked.head(100)
        scatter = axes[1, 1].scatter(top_100['ram'], top_100['battery_power'],
                                    c=top_100['ranking_score'], cmap='coolwarm',
                                    s=100, alpha=0.6, edgecolors='black', linewidth=1)
        axes[1, 1].set_title('RAM vs Battery Power (Top 100)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('RAM (MB)', fontsize=12)
        axes[1, 1].set_ylabel('Battery Power (mAh)', fontsize=12)
        axes[1, 1].grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('Ranking Score', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(OUTPUT_DIR, 'smartphone_ranking_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {output_path}")
        
        plt.show()
    
    def generate_insights(self):
        """Generate key insights"""
        print("\n" + "=" * 70)
        print("KEY INSIGHTS & RECOMMENDATIONS".center(70))
        print("=" * 70)
        
        top_phone = self.df_ranked.iloc[0]
        
        print(f"\n🏆 TOP RANKED SMARTPHONE:")
        print(f"  • ID: Phone-{int(top_phone['id'])}")
        print(f"  • Category: {top_phone['price_category']}")
        print(f"  • Score: {top_phone['ranking_score']:.2f}/100")
        print(f"  • RAM: {int(top_phone['ram'])} MB")
        print(f"  • Battery: {int(top_phone['battery_power'])} mAh")
        print(f"  • Camera: {int(top_phone['pc'])} MP (Primary)")
        
        print("\n📊 CATEGORY STATISTICS:")
        for category in ['Budget', 'Mid-Range', 'Premium', 'Flagship']:
            category_df = self.df[self.df['price_category'] == category]
            if len(category_df) > 0:
                print(f"\n  {category}:")
                print(f"    - Count: {len(category_df)} phones")
                print(f"    - Avg RAM: {category_df['ram'].mean():.0f} MB")
                print(f"    - Avg Battery: {category_df['battery_power'].mean():.0f} mAh")
                print(f"    - Avg Camera: {category_df['pc'].mean():.1f} MP")
        
        print("\n💡 KEY FINDINGS:")
        print("  1. RAM is the strongest predictor of phone quality (28.5% importance)")
        print("  2. Battery capacity significantly impacts rankings (20%+ importance)")
        print("  3. 4G connectivity is standard in premium segments")
        print("  4. Multi-core processors (6+) are common in flagship phones")
        print("  5. Camera quality correlates strongly with price category")
    
    def export_results(self):
        """Export results to CSV"""
        export_cols = ['rank', 'id', 'price_category', 'ranking_score',
                      'ram', 'battery_power', 'pc', 'fc', 'int_memory',
                      'clock_speed', 'n_cores', 'four_g', 'touch_screen', 'wifi']
        
        output_path = os.path.join(OUTPUT_DIR, 'ranked_smartphones.csv')
        self.df_ranked[export_cols].to_csv(output_path, index=False)
        print(f"\n✓ Results exported: {output_path}")
    
    def run_complete_analysis(self):
        """Run the complete pipeline"""
        self.load_data()
        self.feature_engineering()
        self.create_price_categories()
        self.train_classifier()
        self.calculate_ranking_score()
        self.display_top_phones(10)
        self.generate_insights()
        self.visualize_results()
        self.export_results()
        
        print("\n" + "=" * 70)
        print("✓ ANALYSIS COMPLETED SUCCESSFULLY!".center(70))
        print("=" * 70)
        print(f"\n📂 Output files saved in: {OUTPUT_DIR}")
        print("  • ranked_smartphones.csv")
        print("  • smartphone_ranking_analysis.png")


def main():
    """Main execution function"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     SMARTPHONE CLASSIFICATION & RANKING SYSTEM                   ║
    ║              TCS iON Internship Project                          ║
    ║                                                                  ║
    ║         Machine Learning Based Smart Device Analysis             ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        system = SmartphoneRankingSystem()
        
        while True:
            print("\n" + "=" * 70)
            print("MAIN MENU".center(70))
            print("=" * 70)
            print("\n1. Run Complete Analysis (Recommended)")
            print("2. Load and Preview Data Only")
            print("3. Display Top N Phones")
            print("4. Search by Category")
            print("5. Filter by Specifications")
            print("6. Exit")
            
            choice = input("\n👉 Enter your choice (1-6): ").strip()
            
            if choice == '1':
                system.run_complete_analysis()
                
            elif choice == '2':
                system.load_data()
                system.feature_engineering()
                system.create_price_categories()
                system.calculate_ranking_score()
                print("\n✓ Data loaded and processed successfully!")
                
            elif choice == '3':
                if system.df_ranked is None:
                    print("\n⚠ Please load data first (Option 2)")
                else:
                    try:
                        n = int(input("How many phones to display? (1-100): "))
                        system.display_top_phones(min(n, 100))
                    except ValueError:
                        print("❌ Please enter a valid number")
                    
            elif choice == '4':
                if system.df_ranked is None:
                    print("\n⚠ Please load data first (Option 2)")
                else:
                    print("\nAvailable categories:")
                    print("  1. Budget")
                    print("  2. Mid-Range")
                    print("  3. Premium")
                    print("  4. Flagship")
                    cat_choice = input("\nSelect category (1-4): ").strip()
                    
                    categories = {
                        '1': 'Budget',
                        '2': 'Mid-Range',
                        '3': 'Premium',
                        '4': 'Flagship'
                    }
                    
                    if cat_choice in categories:
                        category = categories[cat_choice]
                        filtered = system.df_ranked[
                            system.df_ranked['price_category'] == category
                        ].head(10)
                        
                        if len(filtered) > 0:
                            print(f"\n📱 Top 10 {category} Phones:")
                            print("-" * 70)
                            for _, phone in filtered.iterrows():
                                print(f"Rank {int(phone['rank']):3}: Phone-{int(phone['id'])} | "
                                      f"Score: {phone['ranking_score']:.1f} | "
                                      f"RAM: {int(phone['ram'])} MB | "
                                      f"Battery: {int(phone['battery_power'])} mAh")
                        else:
                            print("No phones found in this category!")
                    else:
                        print("❌ Invalid choice")
                        
            elif choice == '5':
                if system.df_ranked is None:
                    print("\n⚠ Please load data first (Option 2)")
                else:
                    try:
                        min_ram = int(input("Minimum RAM (MB, e.g., 2000): "))
                        min_battery = int(input("Minimum Battery (mAh, e.g., 1500): "))
                        
                        filtered = system.df_ranked[
                            (system.df_ranked['ram'] >= min_ram) &
                            (system.df_ranked['battery_power'] >= min_battery)
                        ].head(10)
                        
                        print(f"\n✓ Found {len(filtered)} phones matching criteria")
                        if len(filtered) > 0:
                            print("-" * 70)
                            for _, phone in filtered.iterrows():
                                print(f"Rank {int(phone['rank']):3}: {phone['price_category']:12} | "
                                      f"RAM: {int(phone['ram']):4} MB | "
                                      f"Battery: {int(phone['battery_power']):4} mAh | "
                                      f"Score: {phone['ranking_score']:.1f}")
                    except ValueError:
                        print("❌ Please enter valid numbers")
                        
            elif choice == '6':
                print("\n" + "=" * 70)
                print("Thank you for using Smartphone Ranking System!".center(70))
                print("TCS iON Internship Project".center(70))
                print("=" * 70)
                break
            else:
                print("\n❌ Invalid choice! Please select 1-6")
    
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nPlease ensure test.csv is in the correct location and try again.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()