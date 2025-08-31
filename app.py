import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
from typing import List, Tuple
import pickle

# === Genetic Algorithm Classes ===

class Individual:
    """Represents a potential team (individual in GA population)"""
    def __init__(self, team_indices: List[int], fitness: float = 0.0):
        self.team_indices = team_indices
        self.fitness = fitness
        self.skill_score = 0.0
        self.personality_diversity = 0.0
        self.skill_balance = 0.0

class GeneticAlgorithm:
    """Genetic Algorithm for team optimization"""
    
    def __init__(self, df, skill_requirements, team_size, personality_requirements=None):
        self.df = df
        self.skill_requirements = skill_requirements
        self.personality_requirements = personality_requirements or {}
        self.team_size = team_size
        self.population_size = 100
        self.generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 10
        
        # Fixed, normalized weights that always sum to 1.0
        self.weights = {
            'skills': 0.5,
            'personality_diversity': 0.3,
            'skill_balance': 0.2
        }
        
        # Filter valid candidates who meet minimum requirements
        self.valid_candidates = self._filter_candidates()
        
    def _filter_candidates(self):
        """Filter candidates who meet minimum skill and personality requirements"""
        filtered_df = self.df.copy()
        
        # Filter by skill requirements (only apply if threshold > 0)
        for skill, min_level in self.skill_requirements.items():
            if skill in filtered_df.columns and min_level > 0:
                # Find candidates with exact level or higher
                filtered_df = filtered_df[filtered_df[skill] >= min_level]
        
        # Filter by individual H/L personality requirements (if specified)
        if self.personality_requirements.get('use_individual_hl', False):
            applied_pattern = self.personality_requirements.get('applied_pattern', '')
            if applied_pattern:
                # Filter to only people matching the applied pattern
                filtered_df = filtered_df[filtered_df['PersonalityType'] == applied_pattern]
        
        return filtered_df.index.tolist()
    
    def _calculate_fitness(self, individual: Individual) -> float:
        """Calculate fitness score for an individual (team) with improved metrics"""
        if len(individual.team_indices) != self.team_size:
            return 0.0
        
        team_data = self.df.loc[individual.team_indices]
        
        if self.personality_requirements.get('use_individual_hl', False):
            applied_pattern = self.personality_requirements.get('applied_pattern', '')
            if applied_pattern:
                matching_members = (team_data['PersonalityType'] == applied_pattern).sum()
                if matching_members < len(team_data):
                    individual.fitness = 0.001
                    return individual.fitness
        
        skill_scores = []
        for skill, min_level in self.skill_requirements.items():
            if skill in team_data.columns and min_level > 0:
                skill_values = team_data[skill].tolist()
                normalized_scores = [score / 6.0 for score in skill_values]
                skill_scores.extend(normalized_scores)
        
        individual.skill_score = np.mean(skill_scores) if skill_scores else 0
        
        if self.personality_requirements.get('use_individual_hl', False):
            applied_pattern = self.personality_requirements.get('applied_pattern', '')
            if applied_pattern:
                pattern_compliance = (team_data['PersonalityType'] == applied_pattern).mean()
                individual.personality_diversity = pattern_compliance
            else:
                individual.personality_diversity = 0.5
        else:
            team_size = len(team_data)
            
            if 'PersonalityType' in team_data.columns and len(team_data) > 0:
                unique_personalities = team_data['PersonalityType'].nunique()
                max_possible_diversity = min(team_size, len(self.df['PersonalityType'].unique()))
                individual.personality_diversity = unique_personalities / max_possible_diversity if max_possible_diversity > 0 else 0
            else:
                individual.personality_diversity = 0.5  # Neutral value
        
        skill_values = []
        for skill, min_level in self.skill_requirements.items():
            if skill in team_data.columns and min_level > 0:
                skill_values.extend(team_data[skill].tolist())
        
        if skill_values and len(skill_values) > 1:
            mean_skill = np.mean(skill_values)
            std_skill = np.std(skill_values)
            
            if mean_skill > 0:
                cv = std_skill / mean_skill
                individual.skill_balance = np.exp(-cv)
            else:
                individual.skill_balance = 0
        else:
            individual.skill_balance = 1 if skill_values else 0
        
        fitness = (
            self.weights['skills'] * individual.skill_score +
            self.weights['personality_diversity'] * individual.personality_diversity +
            self.weights['skill_balance'] * individual.skill_balance
        )
        
        individual.fitness = fitness
        return fitness
    
    def _create_individual(self) -> Individual:
        """Create a random individual (team)"""
        if len(self.valid_candidates) < self.team_size:
            # If not enough valid candidates, use all available data
            candidates = self.df.index.tolist()
        else:
            candidates = self.valid_candidates
        
        team_indices = random.sample(candidates, min(self.team_size, len(candidates)))
        individual = Individual(team_indices)
        self._calculate_fitness(individual)
        return individual
    
    def _initialize_population(self) -> List[Individual]:
        """Initialize the population with random individuals"""
        return [self._create_individual() for _ in range(self.population_size)]
    
    def _selection(self, population: List[Individual]) -> List[Individual]:
        """Tournament selection"""
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        
        return selected
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Order crossover for team formation"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Combine parents and randomly select team_size members
        combined_indices = list(set(parent1.team_indices + parent2.team_indices))
        
        if len(combined_indices) >= self.team_size * 2:
            child1_indices = random.sample(combined_indices, self.team_size)
            remaining = [idx for idx in combined_indices if idx not in child1_indices]
            child2_indices = random.sample(remaining, min(self.team_size, len(remaining)))
            
            # Fill child2 if needed
            if len(child2_indices) < self.team_size:
                additional_needed = self.team_size - len(child2_indices)
                available_indices = [idx for idx in self.valid_candidates if idx not in child2_indices]
                if available_indices:
                    child2_indices.extend(random.sample(available_indices, 
                                                      min(additional_needed, len(available_indices))))
        else:
            # If not enough combined indices, create new random teams
            return self._create_individual(), self._create_individual()
        
        child1 = Individual(child1_indices)
        child2 = Individual(child2_indices)
        
        self._calculate_fitness(child1)
        self._calculate_fitness(child2)
        
        return child1, child2
    
    def _mutate(self, individual: Individual) -> Individual:
        """Mutation: replace random team member"""
        if random.random() > self.mutation_rate:
            return individual
        
        if len(individual.team_indices) == 0:
            return individual
        
        # Replace a random team member
        replace_idx = random.randint(0, len(individual.team_indices) - 1)
        available_candidates = [idx for idx in self.valid_candidates 
                              if idx not in individual.team_indices]
        
        if available_candidates:
            individual.team_indices[replace_idx] = random.choice(available_candidates)
            self._calculate_fitness(individual)
        
        return individual
    
    def analyze_convergence(self, fitness_history, threshold=0.001, consecutive=5):
        """Analyze convergence patterns"""
        # Generations to convergence
        convergence_gen = len(fitness_history)
        for i in range(consecutive, len(fitness_history)):
            recent_changes = []
            for j in range(consecutive):
                if fitness_history[i-j-1] > 0:
                    change = abs(fitness_history[i-j] - fitness_history[i-j-1]) / fitness_history[i-j-1]
                    recent_changes.append(change)
            
            if all(change < threshold for change in recent_changes):
                convergence_gen = i - consecutive
                break
        
        # Convergence rate (first 10 generations)
        early_gens = min(10, len(fitness_history))
        if early_gens > 1:
            convergence_rate = (fitness_history[early_gens-1] - fitness_history[0]) / early_gens
        else:
            convergence_rate = 0
        
        return convergence_gen, convergence_rate
    
    def evolve(self) -> Individual:
        """Main evolution loop"""
        population = self._initialize_population()
        best_fitness_history = []
        
        for generation in range(self.generations):
            # Evaluate fitness
            for individual in population:
                self._calculate_fitness(individual)
            
            # Sort by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)
            best_fitness_history.append(population[0].fitness)
            
            # Create next generation
            next_generation = []
            
            # Elitism: keep best individuals
            next_generation.extend(population[:self.elite_size])
            
            # Generate offspring
            while len(next_generation) < self.population_size:
                # Selection
                selected = self._selection(population)
                
                # Crossover
                for i in range(0, len(selected) - 1, 2):
                    if len(next_generation) >= self.population_size:
                        break
                    
                    child1, child2 = self._crossover(selected[i], selected[i + 1])
                    
                    # Mutation
                    child1 = self._mutate(child1)
                    child2 = self._mutate(child2)
                    
                    next_generation.extend([child1, child2])
            
            population = next_generation[:self.population_size]
        
        # Return best individual
        population.sort(key=lambda x: x.fitness, reverse=True)
        # Add convergence analysis
        convergence_gen, convergence_rate = self.analyze_convergence(best_fitness_history)
        return population[0], best_fitness_history, convergence_gen, convergence_rate

# === Pattern Matching Functions ===

def calculate_pattern_distance(pattern1, pattern2):
    """Calculate the Hamming distance between two personality patterns"""
    if len(pattern1) != len(pattern2):
        return float('inf')
    return sum(c1 != c2 for c1, c2 in zip(pattern1, pattern2))

def find_near_patterns(df, target_pattern, min_candidates=1):
    """Find personality patterns that are close to the target pattern"""
    available_patterns = df['PersonalityType'].unique()
    pattern_matches = []
    
    for pattern in available_patterns:
        distance = calculate_pattern_distance(target_pattern, pattern)
        candidate_count = len(df[df['PersonalityType'] == pattern])
        if candidate_count >= min_candidates:
            pattern_matches.append({
                'pattern': pattern,
                'distance': distance,
                'count': candidate_count,
                'percentage': (candidate_count / len(df)) * 100
            })
    
    # Sort by distance (closest first), then by count (most candidates first)
    pattern_matches.sort(key=lambda x: (x['distance'], -x['count']))
    return pattern_matches

def get_pattern_differences(target_pattern, suggested_pattern):
    """Get detailed differences between target and suggested patterns"""
    traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Emotional Range']
    trait_letters = ['O', 'C', 'E', 'A', 'N']
    
    differences = []
    for i, (trait, letter) in enumerate(zip(traits, trait_letters)):
        target_level = target_pattern[i]
        suggested_level = suggested_pattern[i]
        
        if target_level != suggested_level:
            target_text = "High" if target_level == 'H' else "Low"
            suggested_text = "High" if suggested_level == 'H' else "Low"
            differences.append({
                'trait': trait,
                'letter': letter,
                'target': target_text,
                'suggested': suggested_text,
                'change': f"{target_text} → {suggested_text}"
            })
    
    return differences

# === Team Selection Functions ===

def select_team_ga(df, skill_requirements, team_size, personality_requirements=None):
    """Select team using Genetic Algorithm with fixed weights"""
    ga = GeneticAlgorithm(df, skill_requirements, team_size, personality_requirements)
    best_individual, fitness_history, conv_gen, conv_rate = ga.evolve()  # <-- Updated this line
    
    if best_individual and len(best_individual.team_indices) > 0:
        selected_team = df.loc[best_individual.team_indices].copy()
        return selected_team, best_individual, fitness_history, conv_gen, conv_rate
    else:
        return pd.DataFrame(), None, [], 0, 0

def evaluate_team_ga(df, skill_requirements, team_size, personality_requirements=None, random_trials=20):
    """Evaluate GA team against random selection"""
    ga_team, best_individual, fitness_history, conv_gen, conv_rate = select_team_ga(df, skill_requirements, team_size, personality_requirements)    
    if ga_team.empty:
        return None, None, None, None, None
    
    # Calculate GA team scores (only for required skills)
    required_skills = [skill for skill, level in skill_requirements.items() if level > 0]
    if required_skills:
        ga_scores = ga_team[required_skills].sum(axis=1)
        ga_min, ga_avg, ga_max = ga_scores.min(), ga_scores.mean(), ga_scores.max()
    else:
        ga_min = ga_avg = ga_max = 0.0
    
    # Random team comparison (using same skill filtering)
    rand_mins, rand_avgs, rand_maxs = [], [], []
    for _ in range(random_trials):
        rand_team = df.sample(min(team_size, len(df)))
        if required_skills:
            rand_scores = rand_team[required_skills].sum(axis=1)
            rand_mins.append(rand_scores.min())
            rand_avgs.append(rand_scores.mean())
            rand_maxs.append(rand_scores.max())
        else:
            rand_mins.append(0.0)
            rand_avgs.append(0.0)
            rand_maxs.append(0.0)
    
    rand_min_avg = np.mean(rand_mins)
    rand_avg_avg = np.mean(rand_avgs)
    rand_max_avg = np.mean(rand_maxs)
    
    improvement_over_avg = ((ga_avg - rand_avg_avg) / rand_avg_avg) * 100 if rand_avg_avg != 0 else 0
    
    return ga_team, (rand_min_avg, rand_avg_avg, rand_max_avg), (ga_min, ga_avg, ga_max), improvement_over_avg, (best_individual, fitness_history, conv_gen, conv_rate)# === Role Prediction Functions ===

def load_svm_model(model_path='svm_role_model.pkl'):
    """Load the saved SVM model from .pkl file"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found. Please ensure the model file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_role_with_model(features, model_data):
    """Make role prediction using loaded SVM model"""
    if model_data is None:
        return None
    
    try:
        # Convert and validate input
        input_data = np.array(features)
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        if input_data.shape[1] != len(model_data['feature_names']):
            st.error(f"Expected {len(model_data['feature_names'])} features, got {input_data.shape[1]}")
            return None
        
        # Scale and predict
        input_scaled = model_data['scaler'].transform(input_data)
        prediction = model_data['model'].predict(input_scaled)
        probability = model_data['model'].predict_proba(input_scaled)
        
        return {
            'role': model_data['label_encoder'].inverse_transform(prediction)[0],
            'confidence': float(np.max(probability)),
            'all_probabilities': dict(zip(model_data['target_names'], probability[0])),
            'features_used': model_data['feature_names']
        }
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# === Data Loading Functions ===

@st.cache_data
def load_data():
    """Load data from CSV file and pre-trained model"""
    try:
        # Load the data
        df = pd.read_csv("DeveloperDataset.csv")
        
        # Return the dataframe and a flag indicating it came from CSV
        return df, True
        
    except FileNotFoundError:
        st.sidebar.warning("CSV file not found - using sample data")
        # Create sample data (same as before)
        np.random.seed(42)
        n_samples = 1000
        
        # Technical Skills (0-6 scale)
        technical_skills = [
            'Database Fundamentals', 'Computer Architecture', 'Distributed Computing Systems',
            'Cyber Security', 'Networking', 'Software Development', 'Programming Skills',
            'Project Management', 'Computer Forensics Fundamentals'
        ]
        
        # Create sample data
        data = {}
        for skill in technical_skills:
            data[skill] = np.random.randint(0, 7, n_samples)
        
        # Personality traits (0-1 scale)
        personality_traits = ['Openness', 'Conscientousness', 'Extraversion', 'Agreeableness', 'Emotional_Range']
        for trait in personality_traits:
            data[trait] = np.random.uniform(0, 1, n_samples)
        
        # Values (0-1 scale)
        values = ['Conversation', 'Openness to Change', 'Hedonism', 'Self-enhancement', 'Self-transcendence']
        for value in values:
            data[value] = np.random.uniform(0, 1, n_samples)
        
        # Role
        roles = ['Database Administrator', 'Software Engineer', 'Data Scientist', 'System Administrator', 'Security Analyst']
        data['Role'] = np.random.choice(roles, n_samples)
        
        df = pd.DataFrame(data)
        
        # Create personality types
        def personality_type(row):
            o = 'H' if row['Openness'] > df['Openness'].median() else 'L'
            c = 'H' if row['Conscientousness'] > df['Conscientousness'].median() else 'L'
            e = 'H' if row['Extraversion'] > df['Extraversion'].median() else 'L'
            a = 'H' if row['Agreeableness'] > df['Agreeableness'].median() else 'L'
            n = 'H' if row['Emotional_Range'] > df['Emotional_Range'].median() else 'L'
            return f"{o}{c}{e}{a}{n}"
        
        df['PersonalityType'] = df.apply(personality_type, axis=1)
        
        # Add ID column
        df['ID'] = range(1, len(df) + 1)
        
        return df, False

# === Main Application ===

def main():
    # Page configuration
    st.set_page_config(
        page_title="Project Team Optimizer",
         page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>   
.main-header {
    background: linear-gradient(to bottom, #001f4d 0%, #001f4d 10%, black 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
        border-left: 4px solid #667eea;
    }
    
    .target-pattern {
        background-color: #00000;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        border: 2px solid #28a745;
        margin: 1rem 0;
    }
    
    .suggested-pattern {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        border: 2px solid #ffc107;
        margin: 1rem 0;
    }
    
    .pattern-mismatch {
        background-color: #f8d7da;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
    
    .trait-selector {
        background-color: #00000;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    
    .weights-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Project Team Optimizer</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    if 'data_loaded' not in st.session_state:
        with st.spinner("Loading data..."):
            df, from_csv = load_data()
            
            st.session_state.df = df
            st.session_state.from_csv = from_csv
            st.session_state.data_loaded = True  # Mark data as loaded

    df = st.session_state.df
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Team Optimization", "Role Prediction", "Data Overview"])
    
    with tab1:
        team_optimization_tab(df)
    
    with tab2:
        role_prediction_tab(df)
    
    with tab3:
        data_overview_tab(df)

def team_optimization_tab(df):
    """Team optimization using genetic algorithms"""
    st.header("Project Team Assignment")
    
    # Ensure required columns exist
    if 'PersonalityType' not in df.columns:
        df['PersonalityType'] = df.apply(
            lambda row: ''.join([
                'H' if row['Openness'] > df['Openness'].median() else 'L',
                'H' if row['Conscientousness'] > df['Conscientousness'].median() else 'L',
                'H' if row['Extraversion'] > df['Extraversion'].median() else 'L',
                'H' if row['Agreeableness'] > df['Agreeableness'].median() else 'L',
                'H' if row['Emotional_Range'] > df['Emotional_Range'].median() else 'L',
            ]), axis=1)
    
    # Display data info
    st.sidebar.info(f"Dataset: {len(df)} employees")
    st.sidebar.info(f"Unique Personality Patterns: {df['PersonalityType'].nunique()}")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Team Configuration")
        
        st.markdown("---")
        
        # Skill requirements
        st.subheader("Skill Requirements")
        st.caption("Set minimum skill levels (0 = Not Required)")
        
        requirements = {}
        skill_columns = [col for col in df.columns if col in [
            'Database Fundamentals', 'Computer Architecture', 'Programming Skills', 
            'Project Management', 'AI ML', 'Cyber Security', 'Networking',
            'Software Development', 'Distributed Computing Systems',
            'Computer Forensics Fundamentals'
        ]]
        
        for skill in skill_columns:
            requirements[skill] = st.slider(f'{skill} (0 = Not Required)', 0, 6, 2)
        
        st.markdown("---")
        
        # Personality selection mode
        st.subheader("Personality Selection Mode")
        
      
        personality_mode = st.radio(
            "Choose personality optimization approach:",
            ["ML-Based Diversity", "Individual H/L Traits"],
            help="ML-Based: Uses personality diversity\nIndividual H/L: Target specific High/Low trait combinations",
            key="personality_mode_radio"
        )
        
        personality_requirements = {}
        target_pattern = ""
        applied_pattern = ""
        pattern_suggestion_info = None
        
        if personality_mode == "Individual H/L Traits":
            personality_requirements['use_individual_hl'] = True
            
            st.markdown("""
            <div class="trait-selector">
            <strong>Select H (High) or L (Low) for each personality trait:</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Individual trait selectors
            col1, col2 = st.columns(2)
            
            with col1:
                openness = st.selectbox("Openness", ["H", "L"], help="H = Creative, open to new experiences", key="openness_select")
                conscientiousness = st.selectbox("Conscientiousness", ["H", "L"], help="H = Organized, disciplined", key="conscientiousness_select")
                extraversion = st.selectbox("Extraversion", ["H", "L"], help="H = Outgoing, sociable", key="extraversion_select")
            
            with col2:
                agreeableness = st.selectbox("Agreeableness", ["H", "L"], help="H = Cooperative, trusting", key="agreeableness_select")
                emotional_range = st.selectbox("Emotional Range", ["H", "L"], help="H = Emotionally reactive", key="emotional_range_select")
            
            # Build target pattern
            target_pattern = f"{openness}{conscientiousness}{extraversion}{agreeableness}{emotional_range}"
            personality_requirements['target_pattern'] = target_pattern
            
            # Display the target pattern
            st.markdown(f"""
            <div class="target-pattern">
            Target Pattern: {target_pattern}
            </div>
            """, unsafe_allow_html=True)
            
            # Check pattern availability
            team_size_for_search = st.session_state.get('team_size', 5)
            matching_count = len(df[df['PersonalityType'] == target_pattern])
            
            if matching_count >= team_size_for_search:
                st.success(f"{matching_count} candidates match this pattern")
                applied_pattern = target_pattern
                personality_requirements['applied_pattern'] = applied_pattern
            else:
                st.warning(f"Only {matching_count} candidates match this pattern")
                
                # Find alternatives
                near_patterns = find_near_patterns(df, target_pattern, team_size_for_search)
                
                if near_patterns:
                    closest_pattern = near_patterns[0]
                    applied_pattern = closest_pattern['pattern']
                    
                    st.info(f"Using closest pattern: **{applied_pattern}**")
                    st.info(f"{closest_pattern['count']} candidates available")
                    
                    # Show differences
                    differences = get_pattern_differences(target_pattern, applied_pattern)
                    if differences:
                        st.markdown("**Adaptations:**")
                        for diff in differences[:3]:  # Show first 3 differences
                            st.markdown(f"• **{diff['trait']}**: {diff['change']}")
                    
                    personality_requirements['applied_pattern'] = applied_pattern
                    pattern_suggestion_info = {
                        'target': target_pattern,
                        'applied': applied_pattern,
                        'differences': differences,
                        'distance': closest_pattern['distance'],
                        'candidate_count': closest_pattern['count']
                    }
        
        st.markdown("---")
        
        # Team size
        team_size = st.slider('Team Size', 1, 20, 5, key="team_size_slider")
        st.session_state['team_size'] = team_size
        
        st.markdown("---")
        
        # Optimize button
        optimize_button = st.button("Optimize Team", type="primary", use_container_width=True, key="optimize_button")
    
    # Main content area
    if optimize_button:
        with st.spinner("Evolving optimal team configuration..."):
            team, rand_stats, ga_stats, improvement, ga_results = evaluate_team_ga(
                df, requirements, team_size, 
                personality_requirements if personality_mode == "Individual H/L Traits" else None
            )
        
        if team is None or team.empty:
            st.error("No valid team configuration found. Please adjust your requirements.")
        else:
            best_individual, fitness_history, conv_gen, conv_rate = ga_results            
            # Display results in tabs
            result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
                "Optimal Team", "Performance", "Evolution", "Insights"
            ])
            
            with result_tab1:
                st.subheader("Genetically Optimized Team")
                
                # Team display
                display_columns = ['ID'] if 'ID' in team.columns else []
                
                if personality_mode == "Individual H/L Traits":
                    display_columns.append('PersonalityType')
                    if 'PredictedPersonality' in team.columns:
                        display_columns.append('PredictedPersonality')
                else:
                    if 'PredictedPersonality' in team.columns:
                        display_columns.append('PredictedPersonality')
                    display_columns.append('PersonalityType')
                
                if 'Role' in team.columns:
                    display_columns.append('Role')
                
                # Add required skills
                required_skills = [skill for skill, level in requirements.items() if level > 0]
                display_columns.extend(required_skills)
                
                team_display = team[display_columns].copy()
                team_display.index = [f"Member {i+1}" for i in range(len(team_display))]
                
                st.dataframe(team_display, use_container_width=True)
                
                # Team metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Skills Score", f"{best_individual.skill_score:.2f}")
                
                with col2:
                    if personality_mode == "Individual H/L Traits" and applied_pattern:
                        match_rate = (team['PersonalityType'] == applied_pattern).mean()
                        st.metric("Pattern Match", f"{match_rate:.1%}")
                    else:
                        st.metric("Personality Diversity", f"{best_individual.personality_diversity:.2f}")
                
                with col3:
                    st.metric("Skill Balance", f"{best_individual.skill_balance:.2f}")
                
                with col4:
                    st.metric("Overall Fitness", f"{best_individual.fitness:.3f}")
            
            with result_tab2:
                st.subheader("Performance Analysis")
                
                # Performance comparison
                comparison_data = {
                    'Metric': ['Min Score', 'Avg Score', 'Max Score'],
                    'Random Teams': [f"{rand_stats[0]:.2f}", f"{rand_stats[1]:.2f}", f"{rand_stats[2]:.2f}"],
                    'GA Optimized': [f"{ga_stats[0]:.2f}", f"{ga_stats[1]:.2f}", f"{ga_stats[2]:.2f}"]
                }
                
                st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("GA Improvement", f"{improvement:.1f}%", delta=f"{improvement:.1f}% vs random")
                with col2:
                    st.metric("Team Size", len(team))
            
            with result_tab3:
                st.subheader("Evolution Progress")
                
                if fitness_history:
                    # Plot evolution
                    fig = px.line(
                        x=range(len(fitness_history)), 
                        y=fitness_history,
                        title="Genetic Algorithm Evolution",
                        labels={'x': 'Generation', 'y': 'Best Fitness'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Evolution metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Initial Fitness", f"{fitness_history[0]:.3f}")
                    with col2:
                        st.metric("Final Fitness", f"{fitness_history[-1]:.3f}")
                    with col3:
                        improvement_pct = ((fitness_history[-1] - fitness_history[0]) / fitness_history[0]) * 100 if fitness_history[0] > 0 else 0
                        st.metric("Evolution Gain", f"{improvement_pct:.1f}%")
                    with col4:
                        st.metric("Convergence Generation", conv_gen if conv_gen < len(fitness_history) else "No convergence")
                    with col5:
                        st.metric("Convergence Rate", f"{conv_rate:.4f}")
            
            with result_tab4:
                st.subheader("Team Composition Insights")
                
                # Visualizations based on mode
                if personality_mode == "Individual H/L Traits":
    # For individual H/L mode, show compliance with the applied pattern
                    if applied_pattern:
                        pattern_counts = team['PersonalityType'].value_counts()
                        
                        # Create pie chart
                        fig = px.pie(
                            values=pattern_counts.values,
                            names=pattern_counts.index,
                            title=f"Personality Pattern Distribution (Target: {applied_pattern})",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        
                        # Highlight the target pattern if it exists in the team
                        if applied_pattern in pattern_counts.index:
                            fig.update_traces(
                                marker=dict(
                                    colors=[px.colors.qualitative.Set1[0] if pattern == applied_pattern 
                                        else px.colors.qualitative.Set3[i] 
                                        for i, pattern in enumerate(pattern_counts.index)]
                                )
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show pattern compliance statistics
                        target_count = pattern_counts.get(applied_pattern, 0)
                        total_count = len(team)
                        compliance_rate = (target_count / total_count) * 100 if total_count > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Target Pattern", applied_pattern)
                        with col2:
                            st.metric("Matching Members", f"{target_count}/{total_count}")
                        with col3:
                            st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
                else:
                    # For ML-based diversity mode, show all personality types
                    if 'PersonalityType' in team.columns:
                        personality_counts = team['PersonalityType'].value_counts()
                        
                        # Create pie chart
                        fig = px.pie(
                            values=personality_counts.values,
                            names=personality_counts.index,
                            title="Team Personality Type Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Diversity metrics
                        unique_types = len(personality_counts)
                        total_members = len(team)
                        diversity_score = (unique_types / total_members) * 100 if total_members > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Unique Types", unique_types)
                        with col2:
                            st.metric("Team Size", total_members)
                        with col3:
                            st.metric("Diversity Score", f"{diversity_score:.1f}%")
                        
                        # Show most common personality type
                        if not personality_counts.empty:
                            most_common = personality_counts.index[0]
                            count_most_common = personality_counts.iloc[0]
                            percentage_most_common = (count_most_common / total_members) * 100
                            
                            st.info(f"Most common personality type: **{most_common}** ({count_most_common} members, {percentage_most_common:.1f}%)")
                
                # Skills analysis
                if required_skills:
                    st.markdown("#### Skills Analysis")
                    
                    skills_data = team[required_skills].mean().sort_values(ascending=True)
                    
                    fig = px.bar(
                        x=skills_data.values,
                        y=skills_data.index,
                        orientation='h',
                        title="Average Team Skill Levels",
                        labels={'x': 'Average Skill Level', 'y': 'Skills'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

def role_prediction_tab(df):
    """Role prediction using trained SVM model"""
    st.header("Role Prediction")
    st.markdown("Predict the most suitable role for an employee based on their skills and personality traits using a trained SVM model.")
    
    # Load model
    if 'svm_model' not in st.session_state:
        with st.spinner("Loading SVM model..."):
            st.session_state.svm_model = load_svm_model()
    
    model_data = st.session_state.svm_model
    
    if model_data is None:
        st.error("SVM model not available. Please ensure 'svm_role_model.pkl' is in the application directory.")
        st.info("To generate the model, run the training script for selected features (skills + Big Five traits).")
        return
    
    st.success("SVM model loaded successfully!")
    st.info(f"Model trained on {len(model_data['target_names'])} different roles")
    
    # Two modes: manual input or select from dataset
    input_mode = st.radio(
        "Choose input method:",
        ["Manual Input", "Select from Dataset"],
        horizontal=True
    )
    
    if input_mode == "Manual Input":
        st.subheader("Enter Employee Information")
        
        # Define the exact 14 features used in the selected features model
        skill_features = [
            'Database Fundamentals', 'Computer Architecture', 
            'Distributed Computing Systems', 'Cyber Security', 
            'Networking', 'Software Development', 
            'Programming Skills', 'Project Management', 
            'Computer Forensics Fundamentals'
        ]
        
        personality_features = [
            'Openness to Change', 'Conscientousness', 'Extraversion', 
            'Agreeableness', 'Emotional_Range'
        ]
        
        # Create input form
        with st.form("role_prediction_form"):
            col1, col2 = st.columns(2)
            
            input_values = {}
            
            # Skills section (9 skills)
            with col1:
                st.subheader("Technical Skills")
                st.caption("Rate skills from 0 (No knowledge) to 6 (Expert)")
                
                for skill in skill_features:
                    input_values[skill] = st.slider(
                        skill, 
                        min_value=0, 
                        max_value=6, 
                        value=2, 
                        key=f"skill_{skill}"
                    )
            
            # Big Five personality traits section (5 traits)
            with col2:
                st.subheader("Big Five Personality Traits")
                st.caption("Rate personality traits from 0.0 (Low) to 1.0 (High)")
                
                for trait in personality_features:
                    input_values[trait] = st.slider(
                        trait, 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.5, 
                        step=0.1,
                        key=f"personality_{trait}"
                    )
            
            # Submit button
            submitted = st.form_submit_button("Predict Role", type="primary")
            
            if submitted:
                # Prepare input array in the exact order expected by the model
                input_array = []
                
                # Add skills first (9 features)
                for skill in skill_features:
                    input_array.append(float(input_values[skill]))
                
                # Add personality traits (5 features)  
                for trait in personality_features:
                    input_array.append(float(input_values[trait]))
                
                st.info(f"Input features: {len(input_array)} values (9 skills + 5 personality traits)")
                
                # Make prediction
                prediction = predict_role_with_model(input_array, model_data)
                
                if prediction:
                    st.markdown("---")
                    st.subheader("Prediction Results")
                    
                    # Main result
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Role", prediction['role'])
                    with col2:
                        st.metric("Confidence", f"{prediction['confidence']:.1%}")
                    
                    # Show input summary
                    st.subheader("Input Summary")
                    input_summary = pd.DataFrame({
                        'Feature Type': (['Skill'] * 9) + (['Personality'] * 5),
                        'Feature Name': skill_features + personality_features,
                        'Value': input_array
                    })
                    st.dataframe(input_summary, hide_index=True, use_container_width=True)
                    
                    # Probability breakdown
                    st.subheader("All Role Probabilities")
                    prob_df = pd.DataFrame([
                        {"Role": role, "Probability": f"{prob:.1%}", "Score": prob}
                        for role, prob in sorted(prediction['all_probabilities'].items(), 
                                               key=lambda x: x[1], reverse=True)
                    ])
                    
                    # Create bar chart
                    fig = px.bar(
                        prob_df, 
                        x='Score', 
                        y='Role',
                        orientation='h',
                        title="Role Prediction Probabilities",
                        labels={'Score': 'Probability', 'Role': 'Job Role'}
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display probability table
                    st.dataframe(prob_df[['Role', 'Probability']], hide_index=True, use_container_width=True)
    
    else:  # Select from Dataset
        st.subheader("Select Employee from Dataset")
        
        # Employee selection
        if 'ID' in df.columns:
            employee_options = [f"Employee {row['ID']}" + (f" - {row['Role']}" if 'Role' in df.columns else "") 
                              for idx, row in df.iterrows()]
            selected_idx = st.selectbox("Choose employee:", range(len(employee_options)), 
                                      format_func=lambda x: employee_options[x])
        else:
            selected_idx = st.selectbox("Choose employee:", range(len(df)), 
                                      format_func=lambda x: f"Employee {x+1}" + (f" - {df.iloc[x]['Role']}" if 'Role' in df.columns else ""))
        
        if st.button("Predict Role for Selected Employee", type="primary"):
            # Get selected employee data
            selected_employee = df.iloc[selected_idx]
            
            # Define the exact 14 features used in the model
            required_features = [
                # 9 Skills
                'Database Fundamentals', 'Computer Architecture', 
                'Distributed Computing Systems', 'Cyber Security', 
                'Networking', 'Software Development', 
                'Programming Skills', 'Project Management', 
                'Computer Forensics Fundamentals',
                
                # 5 Big Five personality traits
                'Openness to Change', 'Conscientousness', 'Extraversion', 
                'Agreeableness', 'Emotional_Range'
            ]
            
            # Prepare input array with exact feature order
            input_array = []
            missing_features = []
            
            for feature in required_features:
                if feature in selected_employee.index:
                    input_array.append(float(selected_employee[feature]))
                else:
                    input_array.append(0.0)  # Default value for missing features
                    missing_features.append(feature)
            
            if missing_features:
                st.warning(f"Missing features (using default 0.0): {', '.join(missing_features)}")
            
            # Make prediction
            prediction = predict_role_with_model(input_array, model_data)
            
            if prediction:
                st.markdown("---")
                st.subheader("Prediction Results")
                
                # Show employee info
                st.markdown("#### Selected Employee Information")
                display_cols = ['ID'] if 'ID' in df.columns else []
                if 'Role' in df.columns:
                    display_cols.append('Role')
                if 'PersonalityType' in df.columns:
                    display_cols.append('PersonalityType')
                
                # Add available required features for display
                available_features = [f for f in required_features if f in df.columns][:6]  # Show first 6
                display_cols.extend(available_features)
                
                if display_cols:
                    employee_display = selected_employee[display_cols].to_frame().T
                    st.dataframe(employee_display, hide_index=True, use_container_width=True)
                
                # Show feature summary used for prediction
                st.markdown("#### Features Used for Prediction")
                feature_summary = pd.DataFrame({
                    'Feature Type': (['Skill'] * 9) + (['Big Five Trait'] * 5),
                    'Feature Name': required_features,
                    'Value': input_array,
                    'Available in Data': [f in df.columns for f in required_features]
                })
                st.dataframe(feature_summary, hide_index=True, use_container_width=True)
                
                # Prediction results
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'Role' in selected_employee.index:
                        st.metric("Actual Role", selected_employee['Role'])
                    else:
                        st.metric("Employee ID", selected_employee.get('ID', 'N/A'))
                
                with col2:
                    st.metric("Predicted Role", prediction['role'])
                
                with col3:
                    st.metric("Confidence", f"{prediction['confidence']:.1%}")
                
                # Accuracy check
                if 'Role' in selected_employee.index:
                    is_correct = selected_employee['Role'] == prediction['role']
                    if is_correct:
                        st.success("Prediction matches actual role!")
                    else:
                        st.warning("Prediction differs from actual role")
                
                # Probability breakdown
                st.subheader("All Role Probabilities")
                prob_df = pd.DataFrame([
                    {"Role": role, "Probability": f"{prob:.1%}", "Score": prob}
                    for role, prob in sorted(prediction['all_probabilities'].items(), 
                                           key=lambda x: x[1], reverse=True)
                ])
                
                # Highlight actual role if available
                if 'Role' in selected_employee.index:
                    actual_role = selected_employee['Role']
                    prob_df['Is_Actual'] = prob_df['Role'] == actual_role
                    
                    # Create bar chart with highlighting
                    fig = px.bar(
                        prob_df, 
                        x='Score', 
                        y='Role',
                        orientation='h',
                        title="Role Prediction Probabilities",
                        labels={'Score': 'Probability', 'Role': 'Job Role'},
                        color='Is_Actual',
                        color_discrete_map={True: 'gold', False: 'lightblue'}
                    )
                else:
                    fig = px.bar(
                        prob_df, 
                        x='Score', 
                        y='Role',
                        orientation='h',
                        title="Role Prediction Probabilities",
                        labels={'Score': 'Probability', 'Role': 'Job Role'}
                    )
                
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(prob_df[['Role', 'Probability']], hide_index=True, use_container_width=True)

def data_overview_tab(df):
    """Data overview and statistics"""
    st.header("Dataset Overview & Statistics")
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Employees", len(df))
    
    with col2:
        if 'Role' in df.columns:
            st.metric("Unique Roles", df['Role'].nunique())
        else:
            st.metric("Data Source", "Sample Data" if not st.session_state.from_csv else "CSV File")
    
    with col3:
        if 'PersonalityType' in df.columns:
            st.metric("Personality Patterns", df['PersonalityType'].nunique())
        else:
            st.metric("Personality Patterns", "N/A")
    
    
    
    # Data tabs
    data_tab1, data_tab3 = st.tabs(["Raw Data", "Analysis"])
    
    with data_tab1:
        st.subheader("Raw Dataset")
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Dataset as CSV",
            data=csv,
            file_name="team_optimization_data.csv",
            mime="text/csv"
        )
    
    
    
    with data_tab3:
        st.subheader("Statistical Analysis")
        
        # Correlation analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            st.markdown("#### Correlation Matrix")
            corr_matrix = df[numeric_columns].corr()
            
            fig = px.imshow(corr_matrix, 
                          title="Feature Correlation Matrix",
                          color_continuous_scale="RdBu")
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.markdown("#### Summary Statistics")
        if len(numeric_columns) > 0:
            st.dataframe(df[numeric_columns].describe(), use_container_width=True)
        else:
            st.info("No numeric columns found for statistical summary.")

if __name__ == "__main__":
    main()