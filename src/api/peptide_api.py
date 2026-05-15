"""
API for working with peptides
"""

import logging
import re

logger = logging.getLogger(__name__)

FIND_LINKS_REGEXP = re.compile(r"PEPTIDE\d,\s*PEPTIDE\d,\s*(\d+):R(\d)-(\d+):R(\d)")


def get_cyc_helm_from_monomers(symbols, start, attach_1, end, attach_2):
    """
    Generate a simple HELM string for a cyclic peptide containing the given monomers.
    """
    helm_str = "PEPTIDE1{"
    for symbol in symbols:
        if len(symbol) > 1:
            helm_str += f"[{symbol}]."
        else:
            helm_str += f"{symbol}."
    helm_str = helm_str[:-1]  # remove trailing '.'
    helm_str += "}" + f"$PEPTIDE1,PEPTIDE1,{start + 1}:R{attach_1}-{end + 1}:R{attach_2}$$$V2.0"
    return helm_str


def _find_longest_link(helm):
    matches = re.findall(FIND_LINKS_REGEXP, helm)
    # Find the match spanning the longest range (largest positional gap)
    longest_link = max(matches, key=lambda x: abs(int(x[0]) - int(x[2])))
    return longest_link


def _strip_brackets(monomer):
    if monomer.startswith("[") and monomer.endswith("]"):
        return monomer[1:-1]
    return monomer


def _find_peptide1(helm, include_braces=True):
    """Find the PEPTIDE1{...} section in a HELM string."""
    match = re.search(r"PEPTIDE1\{.*?\}", helm)
    if not match:
        raise ValueError(
            f"Failed to parse cyclic peptide from HELM string: {helm}: "
            "cannot find PEPTIDE1{...} part"
        )
    if include_braces:
        return match.group(0)
    return match.group(0)[len("PEPTIDE1{"):-1]


class CycPeptide:
    """
    Cyclic peptide representation for FWA demonstration purposes.

    !!! For FWA demonstration only. Please use a standard peptide library
    for production stability. Does not handle CHEM{xxx} sections or
    complex branching. !!!
    """

    def __init__(self, helm, verbose=False):
        self.helm = helm

        if "PEPTIDE2" in self.helm:
            raise RuntimeError(
                f"This code only works with single peptide chains; "
                f"PEPTIDE2 found in HELM:\n{helm}"
            )

        pept_str, conn_str, _1, _2, ver_str = helm.split("$")
        if conn_str == "":
            raise ValueError(
                f"Failed to parse cyclic peptide from HELM string: {helm}: "
                "cannot find connection points"
            )

        # Parse monomers from PEPTIDE1{...}
        mono_str = _find_peptide1(pept_str, include_braces=False)
        self.monomers = [_strip_brackets(m) for m in mono_str.split(".")]

        # Store connection and metadata
        self.conn_str = conn_str
        self._1 = _1
        self._2 = _2
        self.version = ver_str

        # Parse cyclization attachment points
        cyc_start, cyc_attach_1, cyc_end, cyc_attach_2 = _find_longest_link(helm)
        self.cyc_start = int(cyc_start) - 1
        self.cyc_end = int(cyc_end) - 1
        self.cyc_attach_1 = int(cyc_attach_1)
        self.cyc_attach_2 = int(cyc_attach_2)

        if verbose:
            print("Created peptide with this sequence:")
            for i, mono in enumerate(self.monomers):
                print(f"{i}-{mono}", end=",")
            print()
            print(
                f"Cyclization from position {self.cyc_start} at R{self.cyc_attach_1} "
                f"to position {self.cyc_end} at R{self.cyc_attach_2}"
            )

    def __len__(self):
        """Return the number of monomers in the peptide."""
        return len(self.monomers)

    def mutate(self, mutations):
        """
        Generate a mutant of this peptide by substituting monomers at given positions.

        Args:
            mutations (dict): Mapping of {position (int): new_monomer (str)}.
                              E.g. {2: 'R', 5: 'W', 10: 'Y'}

        Returns:
            CycPeptide: New peptide with the requested mutations applied.
        """
        mut_monomers = self.monomers[:]
        for pos, symbol in mutations.items():
            mut_monomers[pos] = symbol

        pept_str = "PEPTIDE1{"
        for symbol in mut_monomers:
            if len(symbol) > 1:
                pept_str += f"[{symbol}]."
            else:
                pept_str += f"{symbol}."
        pept_str = pept_str[:-1]  # remove trailing '.'
        pept_str += "}"

        mut_helm = "$".join([pept_str, self.conn_str, self._1, self._2, self.version])
        return CycPeptide(mut_helm, verbose=False)

    def calc_mutations(self, test_peptide):
        """
        Calculate the mutations needed to go from this peptide to another.

        Args:
            test_peptide (CycPeptide): The target peptide to compare against.

        Returns:
            dict: Mapping of {position: new_monomer} for differing positions.

        Raises:
            NotImplementedError: If the two peptides have different lengths.
        """
        if len(self.monomers) != len(test_peptide.monomers):
            raise NotImplementedError(
                "Unable to calculate mutations when insertions/deletions are involved"
            )

        mutations = {}
        for ipos, (thismono, testmono) in enumerate(
            zip(self.monomers, test_peptide.monomers, strict=False)
        ):
            if thismono != testmono:
                mutations[ipos] = testmono
        return mutations
