#!/usr/bin/env python3
"""
Enhanced WATCHDOG Multi-Agent System
Advanced AI/ML Research Paper Processing and Implementation
Focus: RAG Systems, Advanced NLP, AI Agents, MLOps/CI-CD
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EnhancedWatchdogSystem:
    def __init__(self):
        self.artifacts_dir = Path("artifacts")
        self.system_components = {
            "scraper": "enhanced_scraper.py",
            "extractor": "extractor.py",  # Use existing extractor
            "selector": "enhanced_selector.py", 
            "planner": "enhanced_planner.py",
            "manager": "enhanced_manager.py"
        }
        
        # Enhanced focus areas
        self.focus_areas = [
            "RAG Systems & Vector Databases",
            "Advanced NLP & Language Models", 
            "AI Agents & Multi-Agent Systems",
            "MLOps/CI-CD & Deployment",
            "Computer Vision & Neural Networks"
        ]
        
        print(f"\n{'='*80}")
        print("🚀 ENHANCED WATCHDOG MULTI-AGENT SYSTEM")
        print(f"{'='*80}")
        print("🎯 Advanced AI/ML Research Paper Processing and Implementation")
        print("\n🔬 Focus Areas:")
        for i, area in enumerate(self.focus_areas, 1):
            print(f"   {i}. {area}")
        print(f"\n📁 Artifacts Directory: {self.artifacts_dir.absolute()}")
        print(f"🤖 Multi-Agent Architecture: 4 Specialized Coding Agents")
        print(f"{'='*80}")
    
    def check_dependencies(self):
        """Check if all required components exist"""
        print("\n🔍 Checking Enhanced System Dependencies...")
        
        missing_components = []
        for component, filename in self.system_components.items():
            if not Path(filename).exists():
                missing_components.append(filename)
                print(f"❌ Missing: {filename}")
            else:
                print(f"✅ Found: {filename}")
        
        # Check for enhanced coding agents
        enhanced_agents = ["enhanced_coder1.py", "enhanced_coder2.py", "enhanced_coder3.py", "enhanced_coder4.py"]
        for agent in enhanced_agents:
            if not Path(agent).exists():
                missing_components.append(agent)
                print(f"❌ Missing: {agent}")
            else:
                print(f"✅ Found: {agent}")
        
        if missing_components:
            print(f"\n❌ Missing {len(missing_components)} components. Please ensure all files are present.")
            return False
        
        print(f"\n✅ All enhanced system components found!")
        return True
    
    def run_component(self, component_name, component_file, args=None):
        """Run a system component with enhanced error handling"""
        print(f"\n{'='*60}")
        print(f"🚀 Running Enhanced {component_name.upper()}")
        print(f"{'='*60}")
        
        cmd = [sys.executable, component_file]
        if args:
            cmd.extend(args)
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            end_time = time.time()
            
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ Enhanced {component_name} completed successfully in {duration:.1f}s")
                if result.stdout.strip():
                    print("📋 Output:")
                    print(result.stdout)
                return True
            else:
                print(f"❌ Enhanced {component_name} failed with return code {result.returncode}")
                if result.stderr.strip():
                    print("🚨 Error Output:")
                    print(result.stderr)
                if result.stdout.strip():
                    print("📋 Standard Output:")
                    print(result.stdout)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Enhanced {component_name} timed out after 1 hour")
            return False
        except Exception as e:
            print(f"❌ Error running enhanced {component_name}: {e}")
            return False
    
    def run_enhanced_pipeline(self, papers_per_domain=3):
        """Run the complete enhanced pipeline"""
        print(f"\n🚀 Starting Enhanced WATCHDOG Pipeline")
        print(f"🎯 Target: {papers_per_domain} papers per domain")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        pipeline_start = time.time()
        
        # Step 1: Enhanced Scraping
        print(f"\n📡 Step 1: Enhanced arXiv Scraping")
        if not self.run_component("scraper", self.system_components["scraper"]):
            print("❌ Enhanced scraping failed. Stopping pipeline.")
            return False
        
        # Step 2: Text Extraction (using existing extractor)
        print(f"\n📄 Step 2: PDF Text Extraction")
        if not self.run_component("extractor", self.system_components["extractor"]):
            print("❌ Text extraction failed. Stopping pipeline.")
            return False
        
        # Step 3: Enhanced Selection
        print(f"\n🎯 Step 3: Enhanced Paper Selection")
        if not self.run_component("selector", self.system_components["selector"]):
            print("❌ Enhanced selection failed. Stopping pipeline.")
            return False
        
        # Step 4: Enhanced Planning
        print(f"\n📋 Step 4: Enhanced Project Planning")
        if not self.run_component("planner", self.system_components["planner"]):
            print("❌ Enhanced planning failed. Stopping pipeline.")
            return False
        
        # Step 5: Enhanced Implementation
        print(f"\n🤖 Step 5: Enhanced Multi-Agent Implementation")
        
        # Get all planned projects
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
            
            if self.run_component("manager", self.system_components["manager"], [project_name]):
                successful_implementations += 1
                print(f"✅ Successfully implemented: {project_name}")
            else:
                print(f"❌ Failed to implement: {project_name}")
        
        # Pipeline Summary
        pipeline_end = time.time()
        total_duration = pipeline_end - pipeline_start
        
        print(f"\n{'='*80}")
        print("🎉 ENHANCED WATCHDOG PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"⏰ Total Duration: {total_duration/60:.1f} minutes")
        print(f"📊 Projects Found: {len(project_dirs)}")
        print(f"✅ Successful Implementations: {successful_implementations}")
        print(f"❌ Failed Implementations: {len(project_dirs) - successful_implementations}")
        print(f"📈 Success Rate: {(successful_implementations/len(project_dirs))*100:.1f}%")
        print(f"\n🎯 Focus Areas Covered:")
        for area in self.focus_areas:
            print(f"   • {area}")
        print(f"\n📁 Results saved to: {self.artifacts_dir.absolute()}")
        print(f"🚀 Enhanced AI/ML projects ready for deployment!")
        print(f"{'='*80}")
        
        return successful_implementations > 0
    
    def run_individual_component(self, component_name):
        """Run an individual component"""
        if component_name not in self.system_components:
            print(f"❌ Unknown component: {component_name}")
            print(f"Available components: {', '.join(self.system_components.keys())}")
            return False
        
        return self.run_component(component_name, self.system_components[component_name])
    
    def show_system_status(self):
        """Show current system status"""
        print(f"\n{'='*60}")
        print("📊 ENHANCED WATCHDOG SYSTEM STATUS")
        print(f"{'='*60}")
        
        # Check artifacts directory structure
        if self.artifacts_dir.exists():
            print(f"📁 Artifacts Directory: ✅ Exists")
            
            subdirs = ["pdfs", "pdf-txts", "relevant", "structures", "projects"]
            for subdir in subdirs:
                subdir_path = self.artifacts_dir / subdir
                if subdir_path.exists():
                    count = len(list(subdir_path.iterdir()))
                    print(f"   📂 {subdir}: ✅ {count} items")
                else:
                    print(f"   📂 {subdir}: ❌ Not found")
        else:
            print(f"📁 Artifacts Directory: ❌ Not found")
        
        # Check system components
        print(f"\n🔧 System Components:")
        for component, filename in self.system_components.items():
            status = "✅" if Path(filename).exists() else "❌"
            print(f"   {component}: {status} {filename}")
        
        # Check enhanced agents
        print(f"\n🤖 Enhanced Coding Agents:")
        agents = ["enhanced_coder1.py", "enhanced_coder2.py", "enhanced_coder3.py", "enhanced_coder4.py"]
        specializations = [
            "RAG Systems & Vector Databases",
            "Advanced NLP & Language Models", 
            "AI Agents & Multi-Agent Systems",
            "MLOps/CI-CD & Computer Vision"
        ]
        
        for agent, spec in zip(agents, specializations):
            status = "✅" if Path(agent).exists() else "❌"
            print(f"   {agent}: {status} ({spec})")
        
        print(f"{'='*60}")
    
    def cleanup_artifacts(self):
        """Clean up artifacts directory"""
        print(f"\n🧹 Cleaning up artifacts directory...")
        
        if self.artifacts_dir.exists():
            import shutil
            try:
                shutil.rmtree(self.artifacts_dir)
                print(f"✅ Artifacts directory cleaned")
            except Exception as e:
                print(f"❌ Error cleaning artifacts: {e}")
        else:
            print(f"📁 Artifacts directory doesn't exist")


def main():
    """Main execution function"""
    system = EnhancedWatchdogSystem()
    
    if len(sys.argv) < 2:
        print(f"\n📋 Usage: python enhanced_watchdog.py <command> [options]")
        print(f"\nCommands:")
        print(f"  pipeline [papers_per_domain]  - Run complete enhanced pipeline (default: 3 papers)")
        print(f"  scraper                       - Run enhanced scraper only")
        print(f"  extractor                     - Run text extractor only")
        print(f"  selector                      - Run enhanced selector only")
        print(f"  planner                       - Run enhanced planner only")
        print(f"  manager <project_name>        - Run enhanced manager for specific project")
        print(f"  status                        - Show system status")
        print(f"  cleanup                       - Clean up artifacts directory")
        print(f"  check                         - Check system dependencies")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    try:
        if command == "pipeline":
            papers_per_domain = 3
            if len(sys.argv) > 2:
                try:
                    papers_per_domain = int(sys.argv[2])
                except ValueError:
                    print("❌ Invalid papers_per_domain value. Using default: 3")
            
            if not system.check_dependencies():
                sys.exit(1)
            
            success = system.run_enhanced_pipeline(papers_per_domain)
            sys.exit(0 if success else 1)
        
        elif command == "check":
            success = system.check_dependencies()
            sys.exit(0 if success else 1)
        
        elif command == "status":
            system.show_system_status()
            sys.exit(0)
        
        elif command == "cleanup":
            system.cleanup_artifacts()
            sys.exit(0)
        
        elif command in ["scraper", "extractor", "selector", "planner"]:
            if not system.check_dependencies():
                sys.exit(1)
            
            success = system.run_individual_component(command)
            sys.exit(0 if success else 1)
        
        elif command == "manager":
            if len(sys.argv) < 3:
                print("❌ Manager command requires project name")
                print("Usage: python enhanced_watchdog.py manager <project_name>")
                sys.exit(1)
            
            project_name = sys.argv[2]
            success = system.run_component("manager", system.system_components["manager"], [project_name])
            sys.exit(0 if success else 1)
        
        else:
            print(f"❌ Unknown command: {command}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print(f"\n⚠️ Enhanced WATCHDOG interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Enhanced WATCHDOG error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()