import logging
import os
import pandas as pd

from stabddg.unitprod import fetch_uniprot_sequences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntactDataController:
    url = "https://ftp.ebi.ac.uk/pub/databases/intact/current/various/mutations.tsv"

    def __init__(
        self,
        output_dir: str | None = None
    ):
        self.output_dir = output_dir
        self.df_raw: pd.DataFrame = pd.DataFrame()  # Raw input data
        self.df: pd.DataFrame = pd.DataFrame()  # Processed data

    def prepare_data(self) -> None:
        self.download_raw_data()
        self.process_raw_data()
        self.add_uniprot_sequences()

    def download_raw_data(self) -> None:
        logger.info("Downloading IntAct Mutations raw data")
        try:
            self.df_raw = pd.read_csv(
                self.url,
                sep="\t",
                on_bad_lines='warn',
                engine="python",
            )

            if self.output_dir is not None:
                self.df_raw.to_parquet(os.path.join(self.output_dir, "df_intact_mutations_raw.parquet"), index=False)
        except Exception as e:
            logger.error(f"Error downloading IntAct Mutations raw data: {e}")

    def process_raw_data(self) -> None:
        if self.df_raw.empty:
            logger.error("No raw data available to process.")
            return

        logger.info("Processing IntAct Mutations raw data")

        # Regular expression to extract UniProtKB accession numbers
        uniprot_regex = r'(\buniprotkb:[^()]+)(?=\()'

        # Columns to drop
        cols_drop = [
            "PubMedID",
            "Figure legend",
            "Interaction AC"
        ]

        # Columns to keep
        cols_features = [
            "Affected protein AC",
            "Feature type",
            "Feature range(s)",
            "Original sequence",
            "Resulting sequence",
            "Interaction participants"
        ]

        cols_features_dict = {
            x: x.lower().replace(" ", "_").replace(r"(", "").replace(r")", "")
            for x in cols_features
        }

        df_raw = self.df_raw.copy()

        # Removing unused columns
        df_raw = df_raw.drop(
            columns=cols_drop
        )

        # Splitting 'Interaction participants' column into individual rows
        df = df_raw['Interaction participants'].str.split("|").explode().to_frame("participant")
        df["participant_protein"] = (
            df["participant"]
            .str.extract(uniprot_regex)
        )
        df["unique_count"] = df['participant_protein'].groupby(df.index).transform("nunique")

        # Adding feature columns
        df = df.join(
            df_raw[cols_features].rename(
                columns=cols_features_dict
            )
        )
        df.index.name = "group_id"

        # Removing non-unitprod IDs: intact, chebi, etc.
        df = df.dropna(
            subset=["participant_protein", "affected_protein_ac"],
            how="any"
        )
        df = df.loc[
            ~df["participant_protein"].isna(),
        ]
        df = df.loc[
            df["affected_protein_ac"].str.contains("uniprot")
        ]
        df = df.loc[
            df["participant_protein"].str.contains("uniprot")
        ]

        df = df.loc[
            ~(
                (df["unique_count"] > 1) &
                (df["participant_protein"] == df["affected_protein_ac"])
            )
        ]

        df = df.drop_duplicates(
            subset=[
                "participant_protein",
                "affected_protein_ac",
                "feature_ranges",
                "original_sequence",
                "resulting_sequence"
            ]
        )

        # Removing "uniprotkb" prefix from protein IDs
        df["participant_protein"] = df["participant_protein"].str.split(":").str[-1]
        df["affected_protein_ac"] = df["affected_protein_ac"].str.split(":").str[-1]

        # Replacing NA values in 'resulting_sequence' with an empty string
        df["resulting_sequence"] = df["resulting_sequence"].fillna("")

        # Extracting start and end positions of mutations
        df['feature_ranges_start'] = df['feature_ranges'].str.split('-').str[0].astype(int)
        df['feature_ranges_end'] = df['feature_ranges'].str.split('-').str[-1].astype(int)
        df['feature_ranges_start'] -= 1  # Index is 1-based

        df = df.reset_index()

        self.df = df

    def add_uniprot_sequences(self) -> None:
        if self.df.empty:
            logger.error("No data available to fetch sequences.")
            return

        logger.info("Adding UniProt sequences to IntAct Mutations data")

        # Collecting all UniProt codes
        uniprot_codes = pd.concat(
            [
                self.df["participant_protein"],
                self.df["affected_protein_ac"]
            ],
            ignore_index=True
        ).unique().tolist()

        fasta_map = fetch_uniprot_sequences(uniprot_codes, batch_size=100)

        self.df["participant_protein_seq"] = self.df["participant_protein"].map(fasta_map)
        self.df["affected_protein_ac_seq"] = self.df["affected_protein_ac"].map(fasta_map)

        self.df = self.df.dropna(
            subset=["participant_protein_seq", "affected_protein_ac_seq"],
            how="any",
            ignore_index=True
        )
        self.df["affected_protein_ac_seq_mut"] = (
            self.df.apply(
                lambda x: (
                    x["affected_protein_ac_seq"][:x["feature_ranges_start"]] +
                    x["resulting_sequence"] +
                    x["affected_protein_ac_seq"][x["feature_ranges_end"]:]
                ),
                axis=1
            )
        )

        if self.output_dir is not None:
            self.df.to_parquet(os.path.join(self.output_dir, "df_intact_mutations.parquet"), index=False)
