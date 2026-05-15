"""
PyHELM - A class for working with HELM

Copyright 2014, quattro research GmbH - Adapted from 'bcp-discovery-informatics-peptide-sar-api'
"""

import re


class NotationError(Exception):
    """Custom exception for invalid HELM"""


class Polymer:
    """Class representing a polymer chain"""

    def __init__(self, p_type, data="", name=""):
        self.type = p_type
        self.data = data
        self.name = name
        self.monomers = re.split(r"\.", data)

    def __str__(self):
        return f"{self.name}{{{self.data}}}"


# noinspection PyShadowingNames
class Connection:
    """A connection has 3 Attributes: Polymer, Start, Stop; All of them are tuples with two elements

    Example: (Polymer1,Polymer2),(20,'R2'),(3,'R1') or (Polymer1,Polymer2),(10,'pair'),(3,'pair')
    """

    def __init__(self, poly, start, stop):
        self.polymers = poly
        self.start = start
        self.stop = stop

    def validate(self):
        """Validate connection between polymer chains"""
        return (
            len(self.polymers) == 2
            and len(self.start) == 2
            and len(self.stop) == 2
            and len(self.polymers[0].monomers) >= self.start[0]
            and len(self.polymers[1].monomers) >= self.stop[0]
        )

    def __str__(self):
        start = f"{self.start[0]}:{self.start[1]}"
        stop = f"{self.stop[0]}:{self.stop[1]}"
        return f"{self.polymers[0].name},{self.polymers[1].name},{start}-{stop}"


# noinspection PyUnresolvedReferences,RegExpRedundantEscape
class HelmObj:
    """A HelmObj is a complete polymer made of simple polymers, connections and attributes, separated by the '$' sign.

    self.polymers = List of polymer objects
    self.connections = List of connection objects
    self.attributes = List of (Polymer,String)
    self.notation = HELM String
    """

    def __init__(self, helm=None):
        self.polymers = []
        self.connections = []
        self.hydrogen_bonds = []
        self.attributes = []
        self.data = ""
        if helm:
            self.parse_helm(helm)

    def __str__(self):
        poly = [str(p) for p in self.polymers]
        conn = [str(c) for c in self.connections]
        attr = [f"{a[0]}{{{a[1]}:{a[2]}}}" for a in self.attributes]
        hbonds = self.hydrogen_bonds
        return (
            "$".join(["|".join(poly), "|".join(conn), "|".join(hbonds), "|".join(attr)])
            + "$"
        )

    @staticmethod
    def validate_helm(helm):
        """Validate input HELM string"""
        # check $ signs
        sections = re.split(r"\$", helm)
        if len(sections) != 5:
            return False

        # check polymer connections
        first_list = re.findall(
            r"\w+(?=\{)", sections[0]
        )  # all polymers from first element
        second_list = re.findall(
            r"\w+(?=[\{,])", sections[1]
        )  # all polymers from second element
        return all(element in first_list for element in second_list)

    def parse_helm(self, helm):
        """Parse input HELM string into objects of Polymers, Connection and Attributes"""
        if not self.validate_helm(helm):
            raise NotationError("Invalid HELM notation")
        sections = re.split(r"\$", helm)
        self.polymers = self.parse_polymers(sections[0])
        self.connections = self.parse_connections(sections[1], polymers=self.polymers)
        self.hydrogen_bonds = self.parse_hydrogen_bonds(sections[2])
        self.attributes = self.parse_attributes(sections[3])

    @staticmethod
    def parse_polymers(poly_section):
        poly_strings = re.split(r"\|", poly_section)
        polymers = []
        for p in poly_strings:
            p_name = re.match(r"\w+(?=\{)", p)
            if not p_name:
                continue
            p_name_str = p_name.group()
            p_type_match = re.match(r"\D+", p_name_str)
            p_type = p_type_match.group() if p_type_match else ""
            p_data = p[len(p_name_str) + 1 : -1]
            polymers.append(Polymer(p_type, p_data, p_name_str))
        return polymers

    @staticmethod
    def parse_connections(conn_section, polymers):
        connections = []
        conn_strings = re.split(r"\|", conn_section)
        for c in conn_strings:
            if not c:
                continue
            items = re.split(r"[,:-]", c)
            if len(items) != 6:
                raise NotationError("Invalid HELM notation in connection section")
            polys = [poly for poly in polymers if poly.name == items[0]]
            polys.extend([poly for poly in polymers if poly.name == items[1]])
            start = (int(items[2]), items[3])
            stop = (int(items[4]), items[5])
            connections.append(Connection(polys, start, stop))
        return connections

    @staticmethod
    def parse_hydrogen_bonds(hydrogen_bonds_section):
        """Not implemented"""
        return [hydrogen_bonds_section] if hydrogen_bonds_section else []

    @staticmethod
    def parse_attributes(attr_section):
        attributes = []
        attr_arr = re.split(r"\|", attr_section)
        for attr_string in attr_arr:
            p_name = re.match(r"\w+(?=\{)", attr_string)
            if not p_name:
                continue
            p_name_str = p_name.group()
            key, value = attr_string[len(p_name_str) + 1 : -1].split(":")
            attributes.append([p_name_str, key, value])
        return attributes

    def get_polymer_names(self):
        """Get Polymer chain names from a HELM string"""
        return [p.name for p in self.polymers]

    def get_polymers(self):
        """Get Polymer chain name and sequence from a HELM string"""
        return [str(p) for p in self.polymers]

    def get_monomers_from_polymer(self, polymer_name):
        """Get all monomers from a Polymer chain - the sequence"""
        for p in self.polymers:
            if p.name == polymer_name:
                return p.monomers
        return None

    def get_distinct_monomers_from_polymer(self, polymer_name):
        """Return a set of unique monomers in a polymer chain"""
        all_monomers = self.get_monomers_from_polymer(polymer_name)
        return set(all_monomers) if all_monomers else None


def example():
    # example use
    my_helm = HelmObj()
    test_helm = "RNA1{[am6]P.R(C)P.R(U)P.R(U)P.R(G)P.R(A)P.R(G)P.R(G)}|PEPTIDE1{A.C.G.[Ac].K.E.D.K.R}|CHEM1{SMCC}$PEPTIDE1,CHEM1,2:R3-1:R2|RNA1,CHEM1,1:R1-1:R1$$$"  # pylint: disable=line-too-long
    print(test_helm)
    my_helm.parse_helm(test_helm)
    print(my_helm)
    for polymer in my_helm.polymers:
        print(polymer.type, polymer.name)
        print(polymer)
        print("")
    for conn in my_helm.connections:
        print(conn.validate())
        print(
            conn.polymers[0].name
            + str(conn.start)
            + " -> "
            + conn.polymers[1].name
            + str(conn.stop)
        )


if __name__ == "__main__":
    example()
