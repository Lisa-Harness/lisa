"""Kernel seams — host-injection protocols for blob, browser, state, and memory.

These Protocols define the dependency-inversion surface between the
runtime-neutral kernel and host applications. Hosts implement these
protocols and inject concrete instances into execution contexts; the
kernel never imports host code.
"""
