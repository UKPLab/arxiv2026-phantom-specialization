"""
Tier 2: Rule-based closed-class post-processing.

Refines LLM categories for closed-class words using curated word lists:
  VERB       -> VERB_LEXICAL / VERB_AUXILIARY / VERB_MODAL
  FUNCTION   -> FUNCTION_DET / FUNCTION_PREP / FUNCTION_CONJ / FUNCTION_PARTICLE
  PRONOUN    -> PRONOUN_PERSONAL / PRONOUN_POSSESSIVE / PRONOUN_DEMONSTRATIVE /
               PRONOUN_REFLEXIVE / PRONOUN_INTERROGATIVE / PRONOUN_INDEFINITE
"""

import json
import logging
from pathlib import Path
from typing import Dict, Set

import pandas as pd


def load_closed_class_lists(path: Path) -> Dict[str, Set[str]]:
    """Load closed-class word lists for Tier 2 refinement."""
    with open(path) as f:
        raw = json.load(f)
    return {key: set(words) for key, words in raw.items()}


def apply_tier2_refinement(
    results_df: pd.DataFrame,
    closed_lists: Dict[str, Set[str]],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Apply rule-based closed-class refinement (Tier 2).
    Adds 'final_category' column based on 'llm_category' + word lists.
    """
    logger.info("Applying Tier 2 closed-class refinement...")

    df = results_df.copy()
    df["final_category"] = df["llm_category"]

    cleaned_lower = df["cleaned_token"].str.lower().str.strip()

    # --- VERB refinement ---
    verb_mask = df["llm_category"] == "VERB"
    if verb_mask.any():
        aux_mask = verb_mask & cleaned_lower.isin(closed_lists["auxiliary_verbs"])
        modal_mask = verb_mask & cleaned_lower.isin(closed_lists["modal_verbs"])
        lexical_mask = verb_mask & ~aux_mask & ~modal_mask

        df.loc[aux_mask, "final_category"] = "VERB_AUXILIARY"
        df.loc[modal_mask, "final_category"] = "VERB_MODAL"
        df.loc[lexical_mask, "final_category"] = "VERB_LEXICAL"

        logger.info(
            f"  VERB -> LEXICAL: {lexical_mask.sum()}, "
            f"AUXILIARY: {aux_mask.sum()}, MODAL: {modal_mask.sum()}"
        )

    # --- FUNCTION refinement ---
    func_mask = df["llm_category"] == "FUNCTION"
    if func_mask.any():
        det_mask = func_mask & cleaned_lower.isin(closed_lists["determiners"])
        prep_mask = (
            func_mask & cleaned_lower.isin(closed_lists["prepositions"]) & ~det_mask
        )
        conj_mask = (
            func_mask
            & cleaned_lower.isin(closed_lists["conjunctions"])
            & ~det_mask
            & ~prep_mask
        )
        particle_mask = func_mask & ~det_mask & ~prep_mask & ~conj_mask

        df.loc[det_mask, "final_category"] = "FUNCTION_DET"
        df.loc[prep_mask, "final_category"] = "FUNCTION_PREP"
        df.loc[conj_mask, "final_category"] = "FUNCTION_CONJ"
        df.loc[particle_mask, "final_category"] = "FUNCTION_PARTICLE"

        logger.info(
            f"  FUNCTION -> DET: {det_mask.sum()}, PREP: {prep_mask.sum()}, "
            f"CONJ: {conj_mask.sum()}, PARTICLE: {particle_mask.sum()}"
        )

    # --- PRONOUN refinement ---
    pron_mask = df["llm_category"] == "PRONOUN"
    if pron_mask.any():
        personal_mask = pron_mask & cleaned_lower.isin(
            closed_lists["pronouns_personal"]
        )
        possessive_mask = (
            pron_mask
            & cleaned_lower.isin(closed_lists["pronouns_possessive"])
            & ~personal_mask
        )
        demonstrative_mask = (
            pron_mask
            & cleaned_lower.isin(closed_lists["pronouns_demonstrative"])
            & ~personal_mask
            & ~possessive_mask
        )
        reflexive_mask = pron_mask & cleaned_lower.isin(
            closed_lists["pronouns_reflexive"]
        )
        interrogative_mask = (
            pron_mask
            & cleaned_lower.isin(closed_lists["pronouns_interrogative"])
            & ~personal_mask
            & ~possessive_mask
            & ~demonstrative_mask
            & ~reflexive_mask
        )
        indefinite_mask = (
            pron_mask
            & cleaned_lower.isin(closed_lists["pronouns_indefinite"])
            & ~personal_mask
            & ~possessive_mask
            & ~demonstrative_mask
            & ~reflexive_mask
            & ~interrogative_mask
        )
        other_pron_mask = (
            pron_mask
            & ~personal_mask
            & ~possessive_mask
            & ~demonstrative_mask
            & ~reflexive_mask
            & ~interrogative_mask
            & ~indefinite_mask
        )

        df.loc[personal_mask, "final_category"] = "PRONOUN_PERSONAL"
        df.loc[possessive_mask, "final_category"] = "PRONOUN_POSSESSIVE"
        df.loc[demonstrative_mask, "final_category"] = "PRONOUN_DEMONSTRATIVE"
        df.loc[reflexive_mask, "final_category"] = "PRONOUN_REFLEXIVE"
        df.loc[interrogative_mask, "final_category"] = "PRONOUN_INTERROGATIVE"
        df.loc[indefinite_mask, "final_category"] = "PRONOUN_INDEFINITE"

        logger.info(
            f"  PRONOUN -> PERSONAL: {personal_mask.sum()}, "
            f"POSSESSIVE: {possessive_mask.sum()}, "
            f"DEMONSTRATIVE: {demonstrative_mask.sum()}, "
            f"REFLEXIVE: {reflexive_mask.sum()}, "
            f"INTERROGATIVE: {interrogative_mask.sum()}, "
            f"INDEFINITE: {indefinite_mask.sum()}, "
            f"OTHER: {other_pron_mask.sum()}"
        )

    # Log final category distribution
    logger.info("")
    logger.info("Final category distribution:")
    for cat, count in df["final_category"].value_counts().items():
        pct = 100 * count / len(df)
        logger.info(f"  {cat:25s}: {count:6,} ({pct:5.2f}%)")

    return df
