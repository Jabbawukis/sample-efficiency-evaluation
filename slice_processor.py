import argparse
import os
from utility import utility
from sample_efficiency_evaluation.fact_matcher import FactMatcherHybrid, FactMatcherSimple, FactMatcherEntityLinking
import datasets

# Argument parser
parser = argparse.ArgumentParser(description="Process a dataset slice.")
parser.add_argument("--start_idx", type=int, required=True)
parser.add_argument("--end_idx", type=int, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--dataset_name", type=str, default="")
parser.add_argument("--bear_data_path", type=str, required=True)
parser.add_argument("--rel_info_output_dir", type=str, required=True)
parser.add_argument("--matcher_type", type=str, choices=["simple", "hybrid", "entity_linker"], required=True)
parser.add_argument("--entity_linker_model", type=str, required=True)
parser.add_argument("--gpu_id", type=int, required=True)
parser.add_argument("--slice_num", type=int, required=True)
parser.add_argument("--file_index_dir", type=str, required=True)
parser.add_argument("--save_file_content", type=lambda x: x.lower() == 'true', required=True)
parser.add_argument("--read_existing_index", type=lambda x: x.lower() == 'true', required=True)
parser.add_argument("--require_gpu", type=lambda x: x.lower() == 'true', required=True)

args = parser.parse_args()
print(args)

# Set GPU environment
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Create the appropriate FactMatcher instance based on matcher type
def create_matcher():
    if args.matcher_type == "hybrid":
        return FactMatcherHybrid(
            bear_data_path=args.bear_data_path,
            read_existing_index=args.read_existing_index,
            require_gpu=args.require_gpu,
            file_index_dir=args.file_index_dir,
            entity_linker_model=args.entity_linker_model,
            save_file_content=args.save_file_content,
            gpu_id=args.gpu_id
        )
    elif args.matcher_type == "simple":
        return FactMatcherSimple(
            bear_data_path=args.bear_data_path,
            read_existing_index=args.read_existing_index,
            require_gpu=args.require_gpu,
            file_index_dir=args.file_index_dir,
            save_file_content=args.save_file_content
        )
    elif args.matcher_type == "entity_linker":
        return FactMatcherEntityLinking(
            bear_data_path=args.bear_data_path,
            read_existing_index=args.read_existing_index,
            require_gpu=args.require_gpu,
            file_index_dir=args.file_index_dir,
            entity_linker_model=args.entity_linker_model,
            save_file_content=args.save_file_content,
            gpu_id=args.gpu_id
        )
    else:
        raise ValueError(f"Unknown matcher type: {args.matcher_type}")

# Load dataset slice and process
dataset_slice = datasets.load_dataset(args.dataset_path, args.dataset_name, split=f"train[{args.start_idx}:{args.end_idx}]")
fact_matcher = create_matcher()
fact_matcher.index_dataset(dataset_slice, text_key="text")  # Adjust "text" if needed
fact_matcher.close()
fact_matcher.create_fact_statistics()

# Save results
output_path = os.path.join(args.rel_info_output_dir, f"{args.slice_num}_relation_info.json")
utility.save_json_dict(fact_matcher.entity_relation_info_dict, output_path)
