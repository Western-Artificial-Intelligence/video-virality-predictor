"""
Simple Sora Prompt Generator - Proof of Concept
Uses only interpretation.csv and generation_specs.json to create Sora prompts
"""

import json
import pandas as pd
from collections import Counter

class SimpleSoraPromptGenerator:
    
    def __init__(self, interpretation_csv, generation_specs_json):
        """Initialize with your existing files"""
        self.df = pd.read_csv(interpretation_csv)
        
        with open(generation_specs_json, 'r') as f:
            self.specs = json.load(f)
        
        # Extract categories from video_id
        self.df['category'] = self.df['video_id'].str.split(' ').str[0]
        
        print(f"✓ Loaded {len(self.df)} videos")
        print(f"✓ Found {len(self.specs)} cluster specs")
    
    def get_cluster_category(self, cluster_id):
        """Get dominant category for a cluster"""
        cluster_videos = self.df[self.df['cluster'] == cluster_id]
        categories = cluster_videos['category'].value_counts()
        
        # Get top 3 categories
        top_cats = categories.head(3).index.tolist()
        
        return top_cats[0] if top_cats else 'general'
    
    def category_to_content(self, category):
        """Convert category to content description"""
        templates = {
            'recipe': 'a cooking tutorial showing food preparation',
            'tech': 'a tech product review and demonstration',
            'gaming': 'exciting gameplay footage',
            'travel': 'exploring a beautiful destination',
            'fashion': 'a fashion outfit showcase',
            'motivational': 'an inspiring motivational message',
            'educational': 'an educational explanation',
            'funny': 'a humorous comedic skit',
            'music': 'a musical performance',
            'challenge': 'someone attempting a challenge',
            'ai': 'an AI technology demonstration',
            'reaction': 'a reaction video with commentary',
            'sports': 'athletic sports action',
            'meme': 'a funny meme or viral moment',
            'random': 'entertaining short content'
        }
        return templates.get(category, 'engaging short-form content')
    
    def metrics_to_vibe(self, spec):
        """Convert cluster metrics to natural language vibe"""
        motion = spec['base_prompt_components']['motion_profile']
        visual = spec['base_prompt_components']['visual_style']
        audio = spec['base_prompt_components']['audio_profile']
        
        # Camera movement
        if motion['camera_movement'] == 'dynamic':
            camera = 'dynamic handheld camera with active movement'
        elif motion['camera_movement'] == 'moderate':
            camera = 'smooth camera with gentle movements'
        else:
            camera = 'stable camera with minimal movement'
        
        # Pacing from cuts per minute
        cuts = motion['cuts_per_minute']
        if cuts > 90:
            pace = 'very fast-paced editing with rapid cuts'
        elif cuts > 60:
            pace = 'fast-paced editing with quick cuts'
        elif cuts > 30:
            pace = 'moderate pacing'
        else:
            pace = 'slow, contemplative pacing'
        
        # Visual style
        if visual['visual_complexity'] == 'high':
            visuals = 'detailed, visually rich scenes'
        elif visual['visual_complexity'] == 'medium':
            visuals = 'balanced, clear composition'
        else:
            visuals = 'clean, minimal visuals'
        
        # Mood from audio
        if audio['volume_level'] == 'high':
            mood = 'energetic, high-energy atmosphere'
        elif audio['volume_level'] == 'medium':
            mood = 'upbeat, engaging mood'
        else:
            mood = 'calm, subdued tone'
        
        return f"{camera}, {pace}, {visuals}, {mood}"
    
    def generate_prompt(self, cluster_id, variation=0):
        """Generate a Sora prompt for a cluster"""
        
        # Get spec for this cluster
        spec = self.specs[str(cluster_id)]
        
        # Get content category
        category = self.get_cluster_category(cluster_id)
        content = self.category_to_content(category)
        
        # Get vibe from metrics
        vibe = self.metrics_to_vibe(spec)
        
        # Add trend tokens if available
        trend_tokens = spec['trend_tokens']
        style_modifier = ""
        if trend_tokens['enabled'] and trend_tokens['tokens']:
            # Cycle through tokens for variations
            tokens = trend_tokens['tokens']
            if variation < len(tokens):
                style_modifier = f" with {tokens[variation]} aesthetic"
        
        # Build final prompt
        prompt = f"{content}{style_modifier}. Shot with {vibe}. Vertical 1080x1920 format optimized for social media shorts. Professional production quality. Duration: 15-20 seconds."
        
        return {
            'cluster_id': cluster_id,
            'prompt': prompt,
            'category': category,
            'metrics': {
                'motion': spec['base_prompt_components']['motion_profile']['motion_intensity'],
                'cuts_per_min': spec['base_prompt_components']['motion_profile']['cuts_per_minute'],
                'approach': spec['generation_approach']
            }
        }
    
    def generate_all_prompts(self, variations_per_cluster=3):
        """Generate prompts for all clusters"""
        
        all_prompts = {}
        
        for cluster_id in self.specs.keys():
            cluster_prompts = []
            
            for i in range(variations_per_cluster):
                prompt_data = self.generate_prompt(int(cluster_id), variation=i)
                cluster_prompts.append(prompt_data)
            
            all_prompts[f'cluster_{cluster_id}'] = cluster_prompts
        
        return all_prompts
    
    def save_prompts(self, output_file='sora_prompts.json'):
        """Generate and save all prompts"""
        
        prompts = self.generate_all_prompts()
        
        with open(output_file, 'w') as f:
            json.dump(prompts, f, indent=2)
        
        print(f"\n✓ Generated {sum(len(p) for p in prompts.values())} prompts")
        print(f"✓ Saved to {output_file}")
        
        return prompts
    
    def print_summary(self):
        """Print a summary of all clusters"""
        
        print("\n" + "="*70)
        print("CLUSTER SUMMARY")
        print("="*70)
        
        for cluster_id in self.specs.keys():
            spec = self.specs[str(cluster_id)]
            category = self.get_cluster_category(int(cluster_id))
            cluster_size = len(self.df[self.df['cluster'] == int(cluster_id)])
            
            print(f"\nCluster {cluster_id}:")
            print(f"  Videos: {cluster_size}")
            print(f"  Category: {category}")
            print(f"  Approach: {spec['generation_approach']}")
            print(f"  Motion: {spec['base_prompt_components']['motion_profile']['motion_intensity']}/10")
            print(f"  Cuts/min: {spec['base_prompt_components']['motion_profile']['cuts_per_minute']}")
            
            # Show sample prompt
            sample = self.generate_prompt(int(cluster_id))
            print(f"  Sample: {sample['prompt'][:100]}...")
        
        print("\n" + "="*70)


def main():
    """Run the prompt generator"""
    
    print("="*70)
    print("SIMPLE SORA PROMPT GENERATOR - PROOF OF CONCEPT")
    print("="*70)
    
    # Initialize
    generator = SimpleSoraPromptGenerator(
        interpretation_csv='interpretation.csv',
        generation_specs_json='generation_specs.json'
    )
    
    # Print summary
    generator.print_summary()
    
    # Generate and save prompts
    prompts = generator.save_prompts('sora_prompts.json')
    
    # Show examples
    print("\n" + "="*70)
    print("EXAMPLE PROMPTS")
    print("="*70)
    
    for cluster_name, variations in list(prompts.items())[:3]:
        print(f"\n{cluster_name.upper()}:")
        for i, p in enumerate(variations, 1):
            print(f"\n  Variation {i}:")
            print(f"  {p['prompt']}")
    
    print("\n" + "="*70)
    print("✓ READY FOR SORA!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review sora_prompts.json")
    print("2. Send prompts to Sora API")
    print("3. Generate videos!")


if __name__ == "__main__":
    main()
