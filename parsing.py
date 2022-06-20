from pathlib import Path
from typing import List, TypeVar
from xml.etree import ElementTree

from data_model import AdmissablePath, Demand, Link

T = TypeVar("T")


def unwrap(x: T | None) -> T:
    if x is None:
        raise ValueError("unwrap")
    return x


NS = {"ns": "http://sndlib.zib.de/network"}


class NetworkParser:
    path: Path
    xml: ElementTree.ElementTree

    def __init__(self, path: Path) -> None:
        self.path = path
        self.xml = ElementTree.parse(path)

    def links(self) -> List[Link]:
        link_list: List[Link] = []
        for link in self.xml.findall(".//ns:link", NS):
            id = link.attrib["id"]
            source = unwrap(unwrap(link.find("ns:source", NS)).text)
            target = unwrap(unwrap(link.find("ns:target", NS)).text)
            link_list.append(Link(id, source, target))
        return link_list

    def demands(self) -> List[Demand]:
        demand_list: List[Demand] = []
        for demand in self.xml.findall(".//ns:demand", NS):
            admissable_paths: List[AdmissablePath] = []
            for admissable_path in demand.findall(".//ns:admissiblePath", NS):
                admissable_paths.append(
                    AdmissablePath(
                        [
                            unwrap(x.text)
                            for x in admissable_path.findall("ns:linkId", NS)
                        ]
                    )
                )
            id = demand.attrib["id"]
            demand_value = unwrap(unwrap(demand.find("ns:demandValue", NS)).text)
            demand_list.append(Demand(id, float(demand_value), admissable_paths))
        return demand_list
