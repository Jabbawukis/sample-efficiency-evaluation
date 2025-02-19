import os

from utility.utility import load_json_dict


def evaluate_aliases(bear_size):
    abs_path = os.path.abspath(os.path.dirname(__file__)).split("sample_efficiency_evaluation")[0]

    match_dict_no_aliases = load_json_dict(
        f"{abs_path}/sample_efficiency_evaluation_results/fact_matching_results/BEAR-{bear_size}/wikimedia_wikipedia_20231101_en/no_aliases/relation_occurrence_info.json"
    )
    match_dict_aliases = load_json_dict(
        f"{abs_path}/sample_efficiency_evaluation_results/fact_matching_results/BEAR-{bear_size}/wikimedia_wikipedia_20231101_en/relation_occurrence_info.json"
    )

    subj_with_alias_count = {"has_alias": set(), "no_alias": set()}
    object_with_alias_count = {"has_alias": set(), "no_alias": set()}
    matches_overall_count = set()

    for relation_id, facts in match_dict_aliases.items():
        for subj_id, aliases_fact in facts.items():
            if len(aliases_fact["subj_aliases"]) > 0:
                subj_with_alias_count["has_alias"].add(subj_id)
            else:
                subj_with_alias_count["no_alias"].add(subj_id)

            if len(aliases_fact["obj_aliases"]) > 0:
                object_with_alias_count["has_alias"].add(aliases_fact["obj_id"])
            else:
                object_with_alias_count["no_alias"].add(aliases_fact["obj_id"])
            if aliases_fact["occurrences"] > 0:
                matches_overall_count.add((relation_id, subj_id, aliases_fact["obj_id"]))

    percentage_of_subjects_with_alias = len(subj_with_alias_count["has_alias"]) / (
        len(subj_with_alias_count["has_alias"]) + len(subj_with_alias_count["no_alias"])
    )
    print(f"% of subjects with alias: {percentage_of_subjects_with_alias}\n")
    percentage_of_objects_with_alias = len(object_with_alias_count["has_alias"]) / (
        len(object_with_alias_count["has_alias"]) + len(object_with_alias_count["no_alias"])
    )
    print(f"% of objects with alias: {percentage_of_objects_with_alias}\n")

    matches_with_aliases = {"subj_and_obj": set(), "subj": set(), "obj": set()}
    matches_overall = {"with aliases": set(), "no need for alias": set()}
    no_matches = {"with aliases": set(), "without aliases": set()}

    avg_increase_in_matches = {"sum": 0, "count": 0}

    for relation_id, facts in match_dict_aliases.items():
        for subj_id, aliases_fact in facts.items():
            no_aliases_fact = match_dict_no_aliases[relation_id][subj_id]
            # matches where the fact with aliases has more occurrences than the fact without aliases
            if aliases_fact["occurrences"] > no_aliases_fact["occurrences"] > 0:
                if len(aliases_fact["subj_aliases"]) > 0 and len(aliases_fact["obj_aliases"]) > 0:
                    matches_with_aliases["subj_and_obj"].add((relation_id, subj_id, aliases_fact["obj_id"]))
                elif len(aliases_fact["subj_aliases"]) > 0:
                    matches_with_aliases["subj"].add((relation_id, subj_id, aliases_fact["obj_id"]))
                elif len(aliases_fact["obj_aliases"]) > 0:
                    matches_with_aliases["obj"].add((relation_id, subj_id, aliases_fact["obj_id"]))
                else:
                    raise ValueError("Error: matches without aliases in alias dict")
                matches_overall["with aliases"].add((relation_id, subj_id, aliases_fact["obj_id"]))
                avg_increase_in_matches["sum"] += aliases_fact["occurrences"] - no_aliases_fact["occurrences"]
                avg_increase_in_matches["count"] += 1
            # matches where the fact with aliases has occurrences, and the fact without aliases has no occurrences
            elif aliases_fact["occurrences"] > no_aliases_fact["occurrences"] == 0:
                if len(aliases_fact["subj_aliases"]) > 0 and len(aliases_fact["obj_aliases"]) > 0:
                    matches_with_aliases["subj_and_obj"].add((relation_id, subj_id, aliases_fact["obj_id"]))
                elif len(aliases_fact["subj_aliases"]) > 0:
                    matches_with_aliases["subj"].add((relation_id, subj_id, aliases_fact["obj_id"]))
                elif len(aliases_fact["obj_aliases"]) > 0:
                    matches_with_aliases["obj"].add((relation_id, subj_id, aliases_fact["obj_id"]))
                else:
                    raise ValueError("Error: matches without aliases in alias dict")
                matches_overall["with aliases"].add((relation_id, subj_id, aliases_fact["obj_id"]))
                no_matches["without aliases"].add((relation_id, subj_id, no_aliases_fact["obj_id"]))
                avg_increase_in_matches["sum"] += aliases_fact["occurrences"]
                avg_increase_in_matches["count"] += 1
            # matches where the fact without aliases the same number of occurrences as the fact with aliases
            elif aliases_fact["occurrences"] == no_aliases_fact["occurrences"] > 0:
                matches_overall["no need for alias"].add((relation_id, subj_id, no_aliases_fact["obj_id"]))
                avg_increase_in_matches["count"] += 1
            # instances were the fact with and the fact without aliases has no matches
            elif aliases_fact["occurrences"] == no_aliases_fact["occurrences"] == 0:
                no_matches["with aliases"].add((relation_id, subj_id, aliases_fact["obj_id"]))
                no_matches["without aliases"].add((relation_id, subj_id, no_aliases_fact["obj_id"]))
                avg_increase_in_matches["count"] += 1
            # instances were the fact with aliases has fewer occurrences than the fact without aliases (should not happen)
            elif aliases_fact["occurrences"] < no_aliases_fact["occurrences"]:
                raise ValueError("Error: fact with aliases has fewer occurrences than fact without aliases")

    matches_overall_count_as_int = len(matches_overall_count)
    print(f"Overall number of instances with matches: {matches_overall_count_as_int}\n")

    overall_matches_aliases = len(matches_overall["with aliases"])
    overall_matches_no_need_aliases = len(matches_overall["no need for alias"])

    assert overall_matches_aliases + overall_matches_no_need_aliases == matches_overall_count_as_int

    print(f"Number of instances with more matches achieved due to aliases: {overall_matches_aliases}\n")
    print(
        f"Number of instances with matches achieved without the need for aliases: {overall_matches_no_need_aliases}\n"
    )

    non_matches_aliases = len(no_matches["with aliases"])
    non_matches_no_aliases = len(no_matches["without aliases"])
    print(f"Number of instances with no matches with aliases: {non_matches_aliases}\n")
    print(f"Number of instances with no matches without aliases: {non_matches_no_aliases}\n")

    percentage_of_matches_with_subject_and_object_aliases = (
        len(matches_with_aliases["subj_and_obj"]) / overall_matches_aliases
    )
    percentage_of_matches_with_subject_aliases = len(matches_with_aliases["subj"]) / overall_matches_aliases
    percentage_of_matches_with_object_aliases = len(matches_with_aliases["obj"]) / overall_matches_aliases

    assert (
        percentage_of_matches_with_subject_and_object_aliases
        + percentage_of_matches_with_subject_aliases
        + percentage_of_matches_with_object_aliases
    ) == 1.0
    print(
        f"% of instances with more matches having subject and object aliases: {percentage_of_matches_with_subject_and_object_aliases}\n"
    )
    print(
        f"% of instances with more matches having only subject aliases: {percentage_of_matches_with_subject_aliases}\n"
    )
    print(f"% of instances with more matches having only object aliases: {percentage_of_matches_with_object_aliases}\n")

    percentage_of_instances_with_matches_where_aliases_mattered = overall_matches_aliases / (
        overall_matches_aliases + overall_matches_no_need_aliases
    )
    print(
        f"% of instances with more matches due to aliases (over all instances with matches): {percentage_of_instances_with_matches_where_aliases_mattered}\n"
    )

    avg_increase_in_matches = avg_increase_in_matches["sum"] / avg_increase_in_matches["count"]
    print(f"Average increase in matches (per fact) due to aliases: {avg_increase_in_matches}\n")


if __name__ == "__main__":
    bear_sizes = ["big", "small"]
    for size in bear_sizes:
        print(f"BEAR-{size}")
        evaluate_aliases(size)
        print("\n\n")
