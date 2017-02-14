#!/usr/bin/env python3
from gtpinterface import gtpinterface
import sys

def main():
	"""
	Main function, simply sends user input on to the gtp interface and prints
	responses.
	"""
	interface = gtpinterface("Q")
	while True:
		command = input()
		success, response = interface.send_command(command)
		print("= " if success else "? ",response,"\n")
		sys.stdout.flush()

if __name__ == "__main__":
	main()
