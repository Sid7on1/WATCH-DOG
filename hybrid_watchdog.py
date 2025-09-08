#!/usr/bin/env python3
"""
Hybrid WATCHDOG Multi-Agent System
Combines original arXiv-to-code functionality with enhanced AI/ML focus
Supports alternating modes: 1 day original, 2 days enhanced
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class HybridWatchdogSystem:
    def __init__(self):
        self.artifacts_dir = Path("artifacts")
        
        # Original system components
        self.original_components = {
            "scraper": "scraper.py",
            "extractor": "extractor.py",
            "selector": "selector.py", 
            "planner": "planner.py",
            "manager": "manager.py"
        }
        
        # Enhanced system components
        self.enhanced_components = {
            "scraper": "enhanced_scraper.py",
            "extractor": "extractor.py",  # Same extractor for both
            "selector": "enhanced_selector.py", 
            "planner": "enhanced_planner.py",
            "manager": "enhanced_manager.py"
        }
        
        # Mode configuration file
        self.mode_config_file = Path("hybrid_mode_config.json")
        
        print(f"\n{'='*80}")
        print("🔄 HYBRID WATCHDOG MULTI-AGENT SYSTEM")
        print(f"{'='*80}")
        print("🎯 Dual-Mode Operation:")
        print("   📊 ORIGINAL MODE: Broad arXiv paper processing (any domain)")
        print("   🚀 ENHANCED MODE: Advanced AI/ML focus (RAG, NLP, Agents, MLOps)")
        print("\n⏰ Scheduling Options:")
        print("   • Auto-alternating: 1 day original → 2 days enhanced → repeat")
        print("   • Manual mode selection")
        print("   • Force specific mode")
        print(f"{'='*80}")
    
    def load_mode_config(self):
        """Load or create mode configuration"""
        default_config = {
            "current_mode": "original",
            "last_run_date": None,
            "cycle_day": 1,  # Day 1 = original, Day 2-3 = enhanced
            "auto_alternate": True,
            "original_papers_per_domain": 2,
            "enhanced_papers_per_domain": 3
        }
        
        if self.mode_config_file.exists():
            try:
                with open(self.mode_config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"⚠️ Error loading mode config: {e}. Using defaults.")
                return default_config
        else:
            return default_config
    
    def save_mode_config(self, config):
        """Save mode configuration"""
        try:
            with open(self.mode_config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving mode config: {e}")
    
    def determine_current_mode(self, force_mode=None):
        """Determine which mode to run based on schedule or force"""
        config = self.load_mode_config()
        
        if force_mode:
            print(f"🔧 Force mode: {force_mode.upper()}")
            config["current_mode"] = force_mode
            config["last_run_date"] = datetime.now().isoformat()
            self.save_mode_config(config)
            return force_mode, config
        
        if not config["auto_alternate"]:
            print(f"📌 Manual mode: {config['current_mode'].upper()}")
            return config["current_mode"], config
        
        # Auto-alternating logic
        today = datetime.now().date()
        
        if config["last_run_date"]:
            last_run = datetime.fromisoformat(config["last_run_date"]).date()
            days_since_last = (today - last_run).days
            
            if days_since_last >= 1:  # At least one day has passed
                # Advance to next day in cycle
                config["cycle_day"] = (config["cycle_day"] % 3) + 1
                
                if config["cycle_day"] == 1:
                    config["current_mode"] = "original"
                else:  # Day 2 or 3
                    config["current_mode"] = "enhanced"
                
                config["last_run_date"] = datetime.now().isoformat()
                self.save_mode_config(config)
        else:
            # First run
            config["current_mode"] = "original"
            config["cycle_day"] = 1
            config["last_run_date"] = datetime.now().isoformat()
            self.save_mode_config(config)
        
        print(f"🔄 Auto-alternating mode: {config['current_mode'].upper()} (Cycle day {config['cycle_day']}/3)")
        return config["current_mode"], config
    
    def check_dependencies(self, mode):
        """Check if all required components exist for the specified mode"""
        print(f"\n🔍 Checking {mode.upper()} Mode Dependencies...")
        
        components = self.enhanced_components if mode == "enhanced" else self.original_components
        missing_components = []
        
        for component, filename in components.items():
            if not Path(filename).exists():
                missing_components.append(filename)
                print(f"❌ Missing: {filename}")
            else:
                print(f"✅ Found: {filename}")
        
        # Check coding agents based on mode
        if mode == "enhanced":
            agents = ["enhanced_coder1.py", "enhanced_coder2.py", "enhanced_coder3.py", "enhanced_coder4.py"]
            agent_names = ["RAG/Vector Specialist", "NLP/Transformer Specialist", "Agent/RL Specialist", "MLOps/CV Specialist"]
        else:
            agents = ["coder1.py", "coder2.py", "coder3.py", "coder4.py"]
            agent_names = ["General Coder 1", "General Coder 2", "General Coder 3", "General Coder 4"]
        
        for agent, name in zip(agents, agent_names):
            if not Path(agent).exists():
                missing_components.append(agent)
                print(f"❌ Missing: {agent} ({name})")
            else:
                print(f"✅ Found: {agent} ({name})")
        
        if missing_components:
            print(f"\n❌ Missing {len(missing_components)} components for {mode} mode.")
            return False
        
        print(f"\n✅ All {mode} mode components found!")
        return True
    
    def run_component(self, component_name, component_file, args=None, mode="original"):
        """Run a system component with enhanced error handling"""
        print(f"\n{'='*60}")
        print(f"🚀 Running {mode.upper()} {component_name.upper()}")
        print(f"{'='*60}")
        
        cmd = [sys.executable, component_file]
        if args:
            cmd.extend(args)
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            end_time = time.time()
            
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ {mode.upper()} {component_name} completed successfully in {duration:.1f}s")
                if result.stdout.strip():
                    print("📋 Output:")
                    print(result.stdout)
                return True
            else:
                print(f"❌ {mode.upper()} {component_name} failed with return code {result.returncode}")
                if result.stderr.strip():
                    print("🚨 Error Output:")
                    print(result.stderr)
                if result.stdout.strip():
                    print("📋 Standard Output:")
                    print(result.stdout)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {mode.upper()} {component_name} timed out after 1 hour")
            return False
        except Exception as e:
            print(f"❌ Error running {mode.upper()} {component_name}: {e}")
            return False
    
    def run_hybrid_pipeline(self, force_mode=None):
        """Run the hybrid pipeline with automatic mode selection"""
        mode, config = self.determine_current_mode(force_mode)
        
        papers_per_domain = config["enhanced_papers_per_domain"] if mode == "enhanced" else config["original_papers_per_domain"]
        
        print(f"\n🚀 Starting HYBRID WATCHDOG Pipeline")
        print(f"🎯 Mode: {mode.upper()}")
        print(f"📊 Target: {papers_per_domain} papers per domain")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not self.check_dependencies(mode):
            return False
        
        components = self.enhanced_components if mode == "enhanced" else self.original_components
        pipeline_start = time.time()
        
        # Step 1: Scraping
        print(f"\n📡 Step 1: {mode.upper()} arXiv Scraping")
        if not self.run_component("scraper", components["scraper"], mode=mode):
            print(f"❌ {mode.upper()} scraping failed. Stopping pipeline.")
            return False
        
        # Step 2: Text Extraction
        print(f"\n📄 Step 2: PDF Text Extraction")
        if not self.run_component("extractor", components["extractor"], mode=mode):
            print(f"❌ Text extraction failed. Stopping pipeline.")
            return False
        
        # Step 3: Selection
        print(f"\n🎯 Step 3: {mode.upper()} Paper Selection")
        if not self.run_component("selector", components["selector"], mode=mode):
            print(f"❌ {mode.upper()} selection failed. Stopping pipeline.")
            return False
        
        # Step 4: Planning
        print(f"\n📋 Step 4: {mode.upper()} Project Planning")
        if not self.run_component("planner", components["planner"], mode=mode):
            print(f"❌ {mode.upper()} planning failed. Stopping pipeline.")
            return False
        
        # Step 5: Implementation
        print(f"\n🤖 Step 5: {mode.upper()} Multi-Agent Implementation")
        
        structures_dir = self.artifacts_dir / "structures"
        if not structures_dir.exists():
            print("❌ No project structures found. Cannot proceed with implementation.")
            return False
        
        project_dirs = [d for d in structures_dir.iterdir() if d.is_dir()]
        if not project_dirs:
            print("❌ No projects to implement.")
            return False
        
        print(f"📊 Found {len(project_dirs)} projects to implement")
        
        successful_implementations = 0
        
        for project_dir in project_dirs:
            project_name = project_dir.name
            print(f"\n🔧 Implementing project: {project_name}")
            
            if self.run_component("manager", components["manager"], [project_name], mode=mode):
                successful_implementations += 1
                print(f"✅ Successfully implemented: {project_name}")
            else:
                print(f"❌ Failed to implement: {project_name}")
        
        # Pipeline Summary
        pipeline_end = time.time()
        total_duration = pipeline_end - pipeline_start
        
        print(f"\n{'='*80}")
        print(f"🎉 HYBRID WATCHDOG PIPELINE COMPLETE - {mode.upper()} MODE")
        print(f"{'='*80}")
        print(f"⏰ Total Duration: {total_duration/60:.1f} minutes")
        print(f"📊 Projects Found: {len(project_dirs)}")
        print(f"✅ Successful Implementations: {successful_implementations}")
        print(f"❌ Failed Implementations: {len(project_dirs) - successful_implementations}")
        print(f"📈 Success Rate: {(successful_implementations/len(project_dirs))*100:.1f}%")
        
        if mode == "enhanced":
            print(f"\n🎯 Enhanced Focus Areas:")
            print(f"   • RAG Systems & Vector Databases")
            print(f"   • Advanced NLP & Language Models")
            print(f"   • AI Agents & Multi-Agent Systems")
            print(f"   • MLOps/CI-CD & Computer Vision")
        else:
            print(f"\n🎯 Original Broad Coverage:")
            print(f"   • All AI/ML domains from arXiv")
            print(f"   • General-purpose implementations")
            print(f"   • Wider research paper coverage")
        
        print(f"\n📁 Results saved to: {self.artifacts_dir.absolute()}")
        print(f"🔄 Next scheduled mode: {'ENHANCED' if mode == 'original' else 'ORIGINAL (after 2 enhanced days)'}")
        print(f"{'='*80}")
        
        return successful_implementations > 0
    
    def show_hybrid_status(self):
        """Show current hybrid system status"""
        config = self.load_mode_config()
        
        print(f"\n{'='*60}")
        print("📊 HYBRID WATCHDOG SYSTEM STATUS")
        print(f"{'='*60}")
        
        # Current mode info
        print(f"🔄 Current Mode: {config['current_mode'].upper()}")
        print(f"📅 Last Run: {config.get('last_run_date', 'Never')}")
        print(f"🔢 Cycle Day: {config['cycle_day']}/3")
        print(f"⚙️ Auto Alternate: {'✅ Enabled' if config['auto_alternate'] else '❌ Disabled'}")
        
        # Next mode prediction
        if config['auto_alternate']:
            next_cycle_day = (config['cycle_day'] % 3) + 1
            next_mode = "original" if next_cycle_day == 1 else "enhanced"
            print(f"🔮 Next Mode: {next_mode.upper()} (Day {next_cycle_day})")
        
        # Check artifacts directory
        if self.artifacts_dir.exists():
            print(f"\n📁 Artifacts Directory: ✅ Exists")
            
            subdirs = ["pdfs", "pdf-txts", "relevant", "structures", "projects"]
            for subdir in subdirs:
                subdir_path = self.artifacts_dir / subdir
                if subdir_path.exists():
                    count = len(list(subdir_path.iterdir()))
                    print(f"   📂 {subdir}: ✅ {count} items")
                else:
                    print(f"   📂 {subdir}: ❌ Not found")
        else:
            print(f"\n📁 Artifacts Directory: ❌ Not found")
        
        # Check both system components
        print(f"\n🔧 ORIGINAL System Components:")
        for component, filename in self.original_components.items():
            status = "✅" if Path(filename).exists() else "❌"
            print(f"   {component}: {status} {filename}")
        
        print(f"\n🚀 ENHANCED System Components:")
        for component, filename in self.enhanced_components.items():
            status = "✅" if Path(filename).exists() else "❌"
            print(f"   {component}: {status} {filename}")
        
        # Check agents
        print(f"\n🤖 Original Coding Agents:")
        original_agents = ["coder1.py", "coder2.py", "coder3.py", "coder4.py"]
        for agent in original_agents:
            status = "✅" if Path(agent).exists() else "❌"
            print(f"   {agent}: {status}")
        
        print(f"\n🤖 Enhanced Coding Agents:")
        enhanced_agents = ["enhanced_coder1.py", "enhanced_coder2.py", "enhanced_coder3.py", "enhanced_coder4.py"]
        specializations = ["RAG/Vector", "NLP/Transformer", "Agent/RL", "MLOps/CV"]
        for agent, spec in zip(enhanced_agents, specializations):
            status = "✅" if Path(agent).exists() else "❌"
            print(f"   {agent}: {status} ({spec})")
        
        print(f"{'='*60}")
    
    def configure_hybrid_mode(self, auto_alternate=None, original_papers=None, enhanced_papers=None):
        """Configure hybrid mode settings"""
        config = self.load_mode_config()
        
        if auto_alternate is not None:
            config["auto_alternate"] = auto_alternate
            print(f"🔄 Auto-alternating: {'Enabled' if auto_alternate else 'Disabled'}")
        
        if original_papers is not None:
            config["original_papers_per_domain"] = original_papers
            print(f"📊 Original mode papers per domain: {original_papers}")
        
        if enhanced_papers is not None:
            config["enhanced_papers_per_domain"] = enhanced_papers
            print(f"🚀 Enhanced mode papers per domain: {enhanced_papers}")
        
        self.save_mode_config(config)
        print("✅ Configuration saved")


def main():
    """Main execution function"""
    system = HybridWatchdogSystem()
    
    if len(sys.argv) < 2:
        print(f"\n📋 Usage: python hybrid_watchdog.py <command> [options]")
        print(f"\nCommands:")
        print(f"  pipeline [original|enhanced]     - Run hybrid pipeline (auto-selects mode)")
        print(f"  original                         - Force original mode pipeline")
        print(f"  enhanced                         - Force enhanced mode pipeline")
        print(f"  status                           - Show hybrid system status")
        print(f"  configure                        - Configure hybrid settings")
        print(f"    --auto-alternate [true|false]  - Enable/disable auto-alternating")
        print(f"    --original-papers <num>        - Papers per domain for original mode")
        print(f"    --enhanced-papers <num>        - Papers per domain for enhanced mode")
        print(f"  cleanup                          - Clean up artifacts directory")
        print(f"\nExamples:")
        print(f"  python hybrid_watchdog.py pipeline          # Auto-select mode")
        print(f"  python hybrid_watchdog.py original          # Force original mode")
        print(f"  python hybrid_watchdog.py enhanced          # Force enhanced mode")
        print(f"  python hybrid_watchdog.py configure --auto-alternate true --enhanced-papers 4")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    try:
        if command == "pipeline":
            force_mode = None
            if len(sys.argv) > 2:
                force_mode = sys.argv[2].lower()
                if force_mode not in ["original", "enhanced"]:
                    print(f"❌ Invalid mode: {force_mode}. Use 'original' or 'enhanced'")
                    sys.exit(1)
            
            success = system.run_hybrid_pipeline(force_mode)
            sys.exit(0 if success else 1)
        
        elif command == "original":
            success = system.run_hybrid_pipeline("original")
            sys.exit(0 if success else 1)
        
        elif command == "enhanced":
            success = system.run_hybrid_pipeline("enhanced")
            sys.exit(0 if success else 1)
        
        elif command == "status":
            system.show_hybrid_status()
            sys.exit(0)
        
        elif command == "configure":
            # Parse configuration arguments
            auto_alternate = None
            original_papers = None
            enhanced_papers = None
            
            i = 2
            while i < len(sys.argv):
                if sys.argv[i] == "--auto-alternate" and i + 1 < len(sys.argv):
                    auto_alternate = sys.argv[i + 1].lower() == "true"
                    i += 2
                elif sys.argv[i] == "--original-papers" and i + 1 < len(sys.argv):
                    original_papers = int(sys.argv[i + 1])
                    i += 2
                elif sys.argv[i] == "--enhanced-papers" and i + 1 < len(sys.argv):
                    enhanced_papers = int(sys.argv[i + 1])
                    i += 2
                else:
                    i += 1
            
            system.configure_hybrid_mode(auto_alternate, original_papers, enhanced_papers)
            sys.exit(0)
        
        elif command == "cleanup":
            if system.artifacts_dir.exists():
                import shutil
                shutil.rmtree(system.artifacts_dir)
                print("✅ Artifacts directory cleaned")
            else:
                print("📁 Artifacts directory doesn't exist")
            sys.exit(0)
        
        else:
            print(f"❌ Unknown command: {command}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print(f"\n⚠️ Hybrid WATCHDOG interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Hybrid WATCHDOG error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()