import argparse
import os
from utility import utility
from sample_efficiency_evaluation.fact_matcher import FactMatcherEntityLinking, FactMatcherSimple
import datasets

# Argument parser
parser = argparse.ArgumentParser(description="Process a dataset slice.")
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--dataset_name", type=str, default="")
parser.add_argument("--bear_data_path", type=str, required=True)
parser.add_argument("--bear_facts_path", type=str, required=True)
parser.add_argument("--rel_info_output_dir", type=str, required=True)
parser.add_argument("--matcher_type", type=str, choices=["simple", "hybrid", "entity_linker"], required=True)
parser.add_argument("--entity_linker_model", type=str, required=True)
parser.add_argument("--gpu_id", type=int, required=True)
parser.add_argument("--total_slices", type=int, required=True)
parser.add_argument("--slice_num", type=int, required=True)
parser.add_argument("--save_file_content", type=lambda x: x.lower() == "true", required=True)
parser.add_argument("--require_gpu", type=lambda x: x.lower() == "true", required=True)

args = parser.parse_args()

# Set GPU environment
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


# Create the appropriate FactMatcher instance based on matcher type
def create_matcher():
    if args.matcher_type == "entity_linker":
        return FactMatcherEntityLinking(
            bear_data_path=args.bear_data_path,
            bear_facts_path=args.bear_facts_path,
            require_gpu=args.require_gpu,
            entity_linker_model=args.entity_linker_model,
            save_file_content=args.save_file_content,
            gpu_id=args.gpu_id,
        )
    elif args.matcher_type == "simple":
        return FactMatcherSimple(
            bear_data_path=args.bear_data_path,
            bear_facts_path=args.bear_facts_path,
            save_file_content=args.save_file_content,
        )
    else:
        raise ValueError(f"Unknown matcher type: {args.matcher_type}")


data_set_name = args.dataset_name if args.dataset_name != "" else None

full_dataset = datasets.load_dataset(args.dataset_path, data_set_name, split="train")
dataset_len = len(full_dataset)
slice_size = dataset_len // args.total_slices
start_index = args.slice_num * slice_size
end_index = dataset_len + 1 if args.slice_num == args.total_slices - 1 else (args.slice_num + 1) * slice_size

print(
    f"Dataset: {args.dataset_path}"
    f"\nDataset name: {data_set_name}"
    f"\nDataset length: {dataset_len}"
    f"\nSlice size: {slice_size}"
    f"\nStart index: {start_index}"
    f"\nEnd index: {end_index}"
    f"\nTotal slices: {args.total_slices}"
    f"\nSlice number: {args.slice_num + 1}"
    f"\nBear data path: {args.bear_data_path}"
    f"\nBear facts path: {args.bear_facts_path}"
    f"\nMatcher type: {args.matcher_type}"
    f"\nOutput directory: {args.rel_info_output_dir}"
    f"\nEntity linker model: {args.entity_linker_model}"
    f"\nGPU ID: {args.gpu_id}"
    f"\nSave file content: {args.save_file_content}"
    f"\nRequire GPU: {args.require_gpu}"
    "\n"
)

# Load dataset slice and process
dataset_slice = datasets.load_dataset(args.dataset_path, args.dataset_name, split=f"train[{start_index}:{end_index}]")
fact_matcher = create_matcher()

fact_matcher.create_fact_statistics(dataset_slice, text_key="text")

# Save results
relation_info_output = os.path.join(args.rel_info_output_dir, f"{args.slice_num}_relation_info.json")
utility.save_dict_as_json(fact_matcher.entity_relation_occurrence_info_dict, relation_info_output)
