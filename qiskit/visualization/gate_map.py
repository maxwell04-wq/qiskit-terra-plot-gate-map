# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A module for visualizing device coupling maps"""

import math
from typing import List

import numpy as np
from rustworkx.visualization import graphviz_draw

from qiskit.exceptions import QiskitError
from qiskit.utils import optionals as _optionals
from qiskit.providers.exceptions import BackendPropertyError
from .exceptions import VisualizationError
from .utils import matplotlib_close_if_inline


def _get_backend_interface_version(backend):
    backend_interface_version = getattr(backend, "version", None)
    return backend_interface_version


@_optionals.HAS_RUSTWORKX.require_in_call
def plot_gate_map(
    backend,
    label_qubits=True,
    qubit_color=None,
    qubit_labels=None,
    line_color=None,
    filename=None,
):

    """Plots the gate map of a device.

    Args:
        backend (Backend): The backend instance that will be used to plot the device
            gate map.
        label_qubits (bool): Label the qubits.
        qubit_color (list): A list of colors for the qubits
        qubit_labels (list): A list of qubit labels
        line_color (list): A list of colors for each line from coupling_map.
        filename (str): file path to save image to.

    Returns:
        Figure: A Rustoworkx graph figure instance.

    Raises:
        QiskitError: if tried to pass a simulator, or if the backend is None,
            but one of num_qubits, mpl_data, or cmap is None.

    Example:

        .. plot::
           :include-source:

           from qiskit import QuantumCircuit, execute
           from qiskit.providers.fake_provider import FakeVigoV2
           from qiskit.visualization import plot_gate_map

           backend = FakeVigoV2()

           plot_gate_map(backend)
    """

    backend_version = _get_backend_interface_version(backend)
    if backend_version <= 1:
        from qiskit.transpiler.coupling import CouplingMap

        if backend.configuration().simulator:
            raise QiskitError("Requires a device backend, not simulator.")
        config = backend.configuration()
        num_qubits = config.n_qubits
        coupling_map = CouplingMap(config.coupling_map)
    else:
        num_qubits = backend.num_qubits
        coupling_map = backend.coupling_map

    return plot_coupling_map(
        num_qubits,
        coupling_map.get_edges(),
        label_qubits,
        qubit_color,
        qubit_labels,
        line_color,
        filename,
    )


@_optionals.HAS_RUSTWORKX.require_in_call
def plot_coupling_map(
    num_qubits: int,
    coupling_map: List[List[int]],
    label_qubits=True,
    qubit_color=None,
    qubit_labels=None,
    line_color=None,
    filename=None,
):
    """Plots an arbitrary coupling map of qubits (embedded in a plane).

    Args:
        num_qubits (int): The number of qubits defined and plotted.
        coupling_map (List[List[int]]): A list of two-element lists, with entries of each nested
            list being the qubit numbers of the bonds to be plotted.
        label_qubits (bool): Label the qubits.
        qubit_color (list): A list of colors for the qubits
        qubit_labels (list): A list of qubit labels
        line_color (list): A list of colors for each line from coupling_map.
        filename (str): file path to save image to.

    Returns:
        Figure: A rustworkx graph figure showing layout.

    Raises:
        QiskitError: If length of qubit labels does not match number of qubits.

    Example:

        .. plot::
           :include-source:

            from qiskit.visualization import plot_coupling_map

            num_qubits = 8
            coupling_map = [[0, 1], [1, 2], [2, 3], [3, 5], [4, 5], [5, 6], [2, 4], [6, 7]]
            plot_coupling_map(num_qubits, coupling_map)
    """

    from qiskit.transpiler.coupling import CouplingMap

    if qubit_labels is None:
        qubit_labels = [str(node) for node in range(num_qubits)]
    else:
        if len(qubit_labels) != num_qubits:
            raise QiskitError("Length of qubit labels does not equal number of qubits.")
    if not label_qubits:
        qubit_labels = [""] * num_qubits

    # set coloring
    if qubit_color is None:
        qubit_color = ['"#648fff"'] * num_qubits
    if line_color is None:
        line_color = ['"#648fff"'] * len(coupling_map)

    graph = CouplingMap(coupling_map).graph
    for node in graph.node_indices():
        graph[node] = node

    for edge, triple in graph.edge_index_map().items():
        graph.update_edge_by_index(edge, (triple[0], triple[1]))

    def color_node(node):
        out_dict = {
            "label": qubit_labels[node],
            "color": qubit_color[node],
            "fillcolor": qubit_color[node],
            "style": "filled",
            "shape": "circle",
        }
        return out_dict

    def color_edge(edge):
        edge_index = list(graph.edge_list()).index(edge)
        out_dict = {
            "color": line_color[edge_index],
            "fillcolor": line_color[edge_index],
        }
        return out_dict

    fig = graphviz_draw(
        graph, method="neato", node_attr_fn=color_node, edge_attr_fn=color_edge, filename=filename
    )
    return fig


@_optionals.HAS_RUSTWORKX.require_in_call
def plot_circuit_layout(circuit, backend, view="virtual"):
    """Plot the layout of a circuit transpiled for a given
    target backend.

    Args:
        circuit (QuantumCircuit): Input quantum circuit.
        backend (Backend): Target backend.
        view (str): Layout view: either 'virtual' or 'physical'.

    Returns:
        Figure: A Rustworkx graph figure showing layout.

    Raises:
        QiskitError: Invalid view type given.
        VisualizationError: Circuit has no layout attribute.

    Example:
        .. plot::
           :include-source:

            import numpy as np
            from qiskit import QuantumCircuit, transpile
            from qiskit.visualization import plot_circuit_layout
            from qiskit.tools.monitor import job_monitor
            from qiskit.providers.fake_provider import FakeVigoV2

            ghz = QuantumCircuit(3, 3)
            ghz.h(0)
            for idx in range(1,3):
                ghz.cx(0,idx)
            ghz.measure(range(3), range(3))

            backend = FakeVigoV2()
            new_circ_lv3 = transpile(ghz, backend=backend, optimization_level=3)
            plot_circuit_layout(new_circ_lv3, backend)
    """
    if circuit._layout is None:
        raise QiskitError("Circuit has no layout. Perhaps it has not been transpiled.")

    backend_version = _get_backend_interface_version(backend)
    if backend_version <= 1:
        num_qubits = backend.configuration().n_qubits
        cmap = backend.configuration().coupling_map
        cmap_len = len(cmap)
    else:
        num_qubits = backend.num_qubits
        cmap = backend.coupling_map
        cmap_len = cmap.graph.num_edges()

    qubits = []
    qubit_labels = [None] * num_qubits

    bit_locations = {
        bit: {"register": register, "index": index}
        for register in circuit._layout.initial_layout.get_registers()
        for index, bit in enumerate(register)
    }
    for index, qubit in enumerate(circuit._layout.initial_layout.get_virtual_bits()):
        if qubit not in bit_locations:
            bit_locations[qubit] = {"register": None, "index": index}

    if view == "virtual":
        for key, val in circuit._layout.initial_layout.get_virtual_bits().items():
            bit_register = bit_locations[key]["register"]
            if bit_register is None or bit_register.name != "ancilla":
                qubits.append(val)
                qubit_labels[val] = str(bit_locations[key]["index"])

    elif view == "physical":
        for key, val in circuit._layout.initial_layout.get_physical_bits().items():
            bit_register = bit_locations[val]["register"]
            if bit_register is None or bit_register.name != "ancilla":
                qubits.append(key)
                qubit_labels[key] = str(key)

    else:
        raise VisualizationError("Layout view must be 'virtual' or 'physical'.")

    qcolors = ['"#648fff"'] * num_qubits
    for k in qubits:
        qcolors[k] = '"#ff91a4"'

    lcolors = ['"#648fff"'] * cmap_len

    for idx, edge in enumerate(cmap):
        if edge[0] in qubits and edge[1] in qubits:
            lcolors[idx] = '"#ff91a4"'

    fig = plot_gate_map(
        backend,
        qubit_color=qcolors,
        qubit_labels=qubit_labels,
        line_color=lcolors,
    )
    return fig


@_optionals.HAS_MATPLOTLIB.require_in_call
@_optionals.HAS_SEABORN.require_in_call
@_optionals.HAS_RUSTWORKX.require_in_call
def plot_error_map(backend, figsize=(12, 9), show_title=True):
    """Plots the error map of a given backend.

    Args:
        backend (Backend): Given backend.
        figsize (tuple): Figure size in inches.
        show_title (bool): Show the title or not.

    Returns:
        Figure: A Rustworkx graph embedded in a matplotlib figure showing error map.

    Raises:
        VisualizationError: The backend does not provide gate errors for the 'sx' gate.
        MissingOptionalLibraryError: If matplotlib or seaborn is not installed

    Example:
        .. plot::
           :include-source:

            from qiskit import QuantumCircuit, execute
            from qiskit.visualization import plot_error_map
            from qiskit.providers.fake_provider import FakeVigoV2

            backend = FakeVigoV2()
            plot_error_map(backend)
    """
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import gridspec, ticker

    color_map = sns.cubehelix_palette(reverse=True, as_cmap=True)

    backend_version = _get_backend_interface_version(backend)
    if backend_version <= 1:
        backend_name = backend.name()
        num_qubits = backend.configuration().n_qubits
        cmap = backend.configuration().coupling_map
        props = backend.properties()
        props_dict = props.to_dict()
        single_gate_errors = [0] * num_qubits
        read_err = [0] * num_qubits
        cx_errors = []
        # sx error rates
        for gate in props_dict["gates"]:
            if gate["gate"] == "sx":
                _qubit = gate["qubits"][0]
                for param in gate["parameters"]:
                    if param["name"] == "gate_error":
                        single_gate_errors[_qubit] = param["value"]
                        break
                else:
                    raise VisualizationError(
                        f"Backend '{backend}' did not supply an error for the 'sx' gate."
                    )
            for line in cmap:
                for item in props_dict["gates"]:
                    if item["qubits"] == line:
                        cx_errors.append(item["parameters"][0]["value"])
                        break
        for qubit in range(num_qubits):
            try:
                read_err[qubit] = props.readout_error(qubit)
            except BackendPropertyError:
                pass

    else:
        backend_name = backend.name
        num_qubits = backend.num_qubits
        cmap = backend.coupling_map
        two_q_error_map = {}
        single_gate_errors = [0] * num_qubits
        read_err = [0] * num_qubits
        cx_errors = []
        for gate, prop_dict in backend.target.items():
            if prop_dict is None or None in prop_dict:
                continue
            for qargs, inst_props in prop_dict.items():
                if inst_props is None:
                    continue
                if gate == "measure":
                    if inst_props.error is not None:
                        read_err[qargs[0]] = inst_props.error
                elif len(qargs) == 1:
                    if inst_props.error is not None:
                        single_gate_errors[qargs[0]] = max(
                            single_gate_errors[qargs[0]], inst_props.error
                        )
                elif len(qargs) == 2:
                    if inst_props.error is not None:
                        two_q_error_map[qargs] = max(
                            two_q_error_map.get(qargs, 0), inst_props.error
                        )
        if cmap:
            for line in cmap.get_edges():
                err = two_q_error_map.get(tuple(line), 0)
                cx_errors.append(err)

    # Convert to percent
    single_gate_errors = 100 * np.asarray(single_gate_errors)
    avg_1q_err = np.mean(single_gate_errors)

    single_norm = matplotlib.colors.Normalize(
        vmin=min(single_gate_errors), vmax=max(single_gate_errors)
    )
    q_colors = [
        f'"{matplotlib.colors.to_hex(color_map(single_norm(err)))}"' for err in single_gate_errors
    ]

    line_colors = []
    if cmap:

        # Convert to percent
        cx_errors = 100 * np.asarray(cx_errors)
        avg_cx_err = np.mean(cx_errors)

        cx_norm = matplotlib.colors.Normalize(vmin=min(cx_errors), vmax=max(cx_errors))
        line_colors = [
            f'"{matplotlib.colors.to_hex(color_map(cx_norm(err)))}"' for err in cx_errors
        ]

    read_err = 100 * np.asarray(read_err)
    avg_read_err = np.mean(read_err)
    max_read_err = np.max(read_err)

    fig = plt.figure(figsize=figsize)
    gridspec.GridSpec(nrows=2, ncols=3)

    grid_spec = gridspec.GridSpec(
        12, 12, height_ratios=[1] * 11 + [0.5], width_ratios=[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
    )

    left_ax = plt.subplot(grid_spec[2:10, :1])
    main_ax = plt.subplot(grid_spec[:11, 1:11])
    right_ax = plt.subplot(grid_spec[2:10, 11:])
    bleft_ax = plt.subplot(grid_spec[-1, :5])
    if cmap:
        bright_ax = plt.subplot(grid_spec[-1, 7:])

    main_ax.imshow(
        plot_gate_map(
            backend,
            qubit_color=q_colors,
            line_color=line_colors,
        )
    )
    main_ax.axis("off")
    main_ax.set_aspect(1)
    if cmap:
        single_cb = matplotlib.colorbar.ColorbarBase(
            bleft_ax, cmap=color_map, norm=single_norm, orientation="horizontal"
        )
        tick_locator = ticker.MaxNLocator(nbins=5)
        single_cb.locator = tick_locator
        single_cb.update_ticks()
        single_cb.update_ticks()
        bleft_ax.set_title(f"H error rate (%) [Avg. = {round(avg_1q_err, 3)}]")

    if cmap is None:
        bleft_ax.axis("off")
        bleft_ax.set_title(f"H error rate (%) = {round(avg_1q_err, 3)}")

    if cmap:
        cx_cb = matplotlib.colorbar.ColorbarBase(
            bright_ax, cmap=color_map, norm=cx_norm, orientation="horizontal"
        )
        tick_locator = ticker.MaxNLocator(nbins=5)
        cx_cb.locator = tick_locator
        cx_cb.update_ticks()
        bright_ax.set_title(f"CNOT error rate (%) [Avg. = {round(avg_cx_err, 3)}]")

    if num_qubits < 10:
        num_left = num_qubits
        num_right = 0
    else:
        num_left = math.ceil(num_qubits / 2)
        num_right = num_qubits - num_left

    left_ax.barh(range(num_left), read_err[:num_left], align="center", color="#DDBBBA")
    left_ax.axvline(avg_read_err, linestyle="--", color="#212121")
    left_ax.set_yticks(range(num_left))
    left_ax.set_xticks([0, round(avg_read_err, 2), round(max_read_err, 2)])
    left_ax.set_yticklabels([str(kk) for kk in range(num_left)], fontsize=12)
    left_ax.invert_yaxis()
    left_ax.set_title("Readout Error (%)", fontsize=12)

    for spine in left_ax.spines.values():
        spine.set_visible(False)

    if num_right:
        right_ax.barh(
            range(num_left, num_qubits), read_err[num_left:], align="center", color="#DDBBBA"
        )
        right_ax.axvline(avg_read_err, linestyle="--", color="#212121")
        right_ax.set_yticks(range(num_left, num_qubits))
        right_ax.set_xticks([0, round(avg_read_err, 2), round(max_read_err, 2)])
        right_ax.set_yticklabels([str(kk) for kk in range(num_left, num_qubits)], fontsize=12)
        right_ax.invert_yaxis()
        right_ax.invert_xaxis()
        right_ax.yaxis.set_label_position("right")
        right_ax.yaxis.tick_right()
        right_ax.set_title("Readout Error (%)", fontsize=12)
    else:
        right_ax.axis("off")

    for spine in right_ax.spines.values():
        spine.set_visible(False)

    if show_title:
        fig.suptitle(f"{backend_name} Error Map", fontsize=24, y=0.9)
    matplotlib_close_if_inline(fig)
    return fig
