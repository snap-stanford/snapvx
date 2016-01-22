snapvx
=====================

**The description and documentation of SnapVX are at [snap.stanford.edu/snapvx](http://snap.stanford.edu/snapvx/).**

#### INSTALLATION
1. Download the latest release [here](http://snap.stanford.edu/snapvx/release/).
2. Uncompress the `snapvx-*.tar.gz` file.

        tar zxvf snapvx-0.3.tar.gz

3. Run `setup.py install` from inside the top-level directory.

        cd snapvx-0.3
        sudo python setup.py install

4. Test the installation. The included unit tests will test basic functionality to ensure that SnapVX and its dependencies are working correctly.

        cd Tests
        python snapvx_test.py

Note: to run SnapVX locally, without installing it system-wide, just copy **snapvx.py** to your working directory.

#### USAGE
SnapVX is used by simply writing
```
import snapvx
```
in the Python file. See the included `Examples/` directory for use cases and syntax.

### CONTACT
Please file bug reports at [github.com/snap-stanford/snapvx](https://github.com/snap-stanford/snapvx). For any other questions, comments, or concerns, please contact [David Hallac](http://www.stanford.edu/~hallac/).
