snapvx
=====================

**The description and documentation of SnapVX are at [snap.stanford.edu/snapvx](http://snap.stanford.edu/snapvx/).**

#### IMPORTANT LINKS
- Source code repo: https://github.com/snap-stanford/snapvx
- HTML User Documentation: http://snap.stanford.edu/snapvx/doc/index.html
- PDF Developer documentation: https://github.com/snap-stanford/snapvx/developerDoc/Documentation.pdf
- Issue Tracker: https://github.com/snap-stanford/snapvx/issues
- Mailing List:

#### DEPENDENCIES
snapvx is tested to work under Python 2.7.

The required dependencies to build the software are NumPy >= 1.6.1, Scipy >= 0.9, CVXPY >= 0.3.9, Snap.py >= 1.2 and a working C/C++ compiler

#### INSTALLATION
1. Download the latest release [here](http://snap.stanford.edu/snapvx/release/).
2. Uncompress the `snapvx-*.tar.gz` file.

        tar zxvf snapvx-0.3.tar.gz

3. Run `setup.py install` from inside the top-level directory.

        cd snapvx-0.3
        sudo python setup.py install

4. Test the installation. The included unit tests will test basic functionality to ensure that SnapVX and its dependencies are working correctly.

        cd Tests
        chmod u+x test_script.sh
        ./test_script.sh

Note: to run SnapVX locally, without installing it system-wide, just run setup.py with the --user flag.

#### USAGE
SnapVX is used by simply writing
```
import snapvx
```
in the Python file. See the included `Examples/` directory for use cases and syntax.

#### DEVELOPMENT
Please consult the developer doc for ways to contribute code, documentation, test cases and general improvments to snapvx

### CONTACT
Please file bug reports at [github.com/snap-stanford/snapvx](https://github.com/snap-stanford/snapvx). For any other questions, comments, or concerns, please contact [David Hallac](http://www.stanford.edu/~hallac/).
