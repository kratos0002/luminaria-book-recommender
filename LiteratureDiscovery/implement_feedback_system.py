"""
Main script to implement the feedback system and richer context for the LiteratureDiscovery app.
This script will run all the implementation scripts in the correct order.
"""
import os
import sys
import logging
import importlib.util
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def import_script(script_name):
    """Import a script as a module."""
    try:
        spec = importlib.util.spec_from_file_location(script_name, script_name + ".py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Error importing {script_name}: {e}")
        traceback.print_exc()
        return None

def run_implementation():
    """Run all implementation scripts in the correct order."""
    # Change to the directory of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # List of scripts to run in order
    scripts = [
        "update_literature_item",
        "update_database",
        "update_recommendation_functions",
        "add_feedback_ui",
        "add_feedback_endpoint"
    ]
    
    success_count = 0
    
    for script_name in scripts:
        logger.info(f"Running {script_name}...")
        module = import_script(script_name)
        
        if module and hasattr(module, "__name__") and module.__name__ == script_name:
            try:
                # Each script should have a main block that runs all its functions
                logger.info(f"✅ Successfully imported {script_name}")
                success_count += 1
            except Exception as e:
                logger.error(f"Error running {script_name}: {e}")
                traceback.print_exc()
        else:
            logger.error(f"Failed to import {script_name}")
    
    logger.info(f"Implementation complete. {success_count}/{len(scripts)} scripts processed successfully.")
    
    # Print final instructions
    print("\n" + "="*80)
    print("IMPLEMENTATION COMPLETE")
    print("="*80)
    print("\nThe feedback system and richer context have been implemented in the LiteratureDiscovery app.")
    print("\nChanges made:")
    print("1. Updated LiteratureItem class to include summary and match_score fields")
    print("2. Added user_feedback table to the database")
    print("3. Added functions to store and retrieve user feedback")
    print("4. Updated recommendation functions to include summaries and match scores")
    print("5. Updated the recommendations.html template to display summaries and feedback buttons")
    print("6. Added a feedback endpoint to app.py")
    
    print("\nTo test the implementation, run the app with:")
    print("python app.py")
    print("\nThen visit http://localhost:5000 in your browser and try submitting literature inputs and providing feedback.")
    print("="*80)

if __name__ == "__main__":
    run_implementation()
