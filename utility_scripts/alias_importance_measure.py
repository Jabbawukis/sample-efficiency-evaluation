from utility.utility import load_json_dict

match_dict_no_aliases = load_json_dict(
    "../fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en/no_aliases/relation_occurrence_info.json"
)
match_dict_aliases = load_json_dict(
    "../fact_matching_results/BEAR-big/wikimedia_wikipedia_20231101_en/relation_occurrence_info.json"
)


subj_with_alias_count = {"has_alias": set(), "no_alias": set()}
object_with_alias_count = {"has_alias": set(), "no_alias": set()}

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

percentage_of_subjects_with_alias = len(subj_with_alias_count["has_alias"]) / (
    len(subj_with_alias_count["has_alias"]) + len(subj_with_alias_count["no_alias"])
)
print(f"Percentage of subjects with alias: {percentage_of_subjects_with_alias}\n")
percentage_of_objects_with_alias = len(object_with_alias_count["has_alias"]) / (
    len(object_with_alias_count["has_alias"]) + len(object_with_alias_count["no_alias"])
)
print(f"Percentage of objects with alias: {percentage_of_objects_with_alias}\n")


matches_with_aliases = {"subj_and_obj": set(), "subj": set(), "obj": set(), "none": set()}
matches_overall = {"with aliases": set(), "without aliases": set()}
no_matches = {"with aliases": set(), "without aliases": set()}

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
                print(f"Error: matches without aliases in alias dict")
            matches_overall["with aliases"].add((relation_id, subj_id, aliases_fact["obj_id"]))
            matches_overall["without aliases"].add((relation_id, subj_id, no_aliases_fact["obj_id"]))
        # matches where the fact with aliases has occurrences, and the fact without aliases has no occurrences
        elif aliases_fact["occurrences"] > no_aliases_fact["occurrences"] == 0:
            if len(aliases_fact["subj_aliases"]) > 0 and len(aliases_fact["obj_aliases"]) > 0:
                matches_with_aliases["subj_and_obj"].add((relation_id, subj_id, aliases_fact["obj_id"]))
            elif len(aliases_fact["subj_aliases"]) > 0:
                matches_with_aliases["subj"].add((relation_id, subj_id, aliases_fact["obj_id"]))
            elif len(aliases_fact["obj_aliases"]) > 0:
                matches_with_aliases["obj"].add((relation_id, subj_id, aliases_fact["obj_id"]))
            else:
                print(f"Error: matches without aliases in alias dict")
            matches_overall["with aliases"].add((relation_id, subj_id, aliases_fact["obj_id"]))
            no_matches["without aliases"].add((relation_id, subj_id, no_aliases_fact["obj_id"]))
        # matches where the fact without aliases the same number of occurrences as the fact with aliases
        elif aliases_fact["occurrences"] == no_aliases_fact["occurrences"] > 0:
            matches_with_aliases["none"].add((relation_id, subj_id, aliases_fact["obj_id"]))
            matches_overall["with aliases"].add((relation_id, subj_id, aliases_fact["obj_id"]))
            matches_overall["without aliases"].add((relation_id, subj_id, no_aliases_fact["obj_id"]))
        # instances were the fact with and the fact without aliases has no matches
        elif aliases_fact["occurrences"] == no_aliases_fact["occurrences"] == 0:
            no_matches["with aliases"].add((relation_id, subj_id, aliases_fact["obj_id"]))
            no_matches["without aliases"].add((relation_id, subj_id, no_aliases_fact["obj_id"]))
        # instances were the fact with aliases has fewer occurrences than the fact without aliases (should not happen)
        elif aliases_fact["occurrences"] < no_aliases_fact["occurrences"]:
            print(f"Error")

overall_matches_aliases = len(matches_overall["with aliases"])
overall_matches_no_aliases = len(matches_overall["without aliases"])
print(f"Number of matches overall achieved with aliases: {overall_matches_aliases}\n")
print(f"Number of matches overall achieved without the need of aliases: {overall_matches_no_aliases}\n")

non_matches_aliases = len(no_matches["with aliases"])
non_matches_no_aliases = len(no_matches["without aliases"])
print(f"Number of non-matches with aliases: {non_matches_aliases}\n")
print(f"Number of non-matches without aliases: {non_matches_no_aliases}\n")

percentage_of_matches_with_subject_and_object_aliases = (
    len(matches_with_aliases["subj_and_obj"]) / overall_matches_aliases
)
percentage_of_matches_with_subject_aliases = len(matches_with_aliases["subj"]) / overall_matches_aliases
percentage_of_matches_with_object_aliases = len(matches_with_aliases["obj"]) / overall_matches_aliases
percentage_of_matches_with_no_aliases = len(matches_with_aliases["none"]) / overall_matches_aliases

assert (
    percentage_of_matches_with_subject_and_object_aliases
    + percentage_of_matches_with_subject_aliases
    + percentage_of_matches_with_object_aliases
    + percentage_of_matches_with_no_aliases
) == 1.0
print(
    f"Percentage of matches with subject and object aliases: {percentage_of_matches_with_subject_and_object_aliases}\n"
)
print(f"Percentage of matches with only subject aliases: {percentage_of_matches_with_subject_aliases}\n")
print(f"Percentage of matches with only object aliases: {percentage_of_matches_with_object_aliases}\n")
print(f"Percentage of matches where aliases did not matter: {percentage_of_matches_with_no_aliases}\n")

assert overall_matches_aliases > overall_matches_no_aliases
increase_of_matches_due_to_aliases = overall_matches_aliases / overall_matches_no_aliases
print(f"Increase of matches due to aliases: {increase_of_matches_due_to_aliases}\n")

assert overall_matches_aliases + non_matches_aliases == overall_matches_no_aliases + non_matches_no_aliases
